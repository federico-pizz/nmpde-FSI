#include "FSI.hpp"
#include <filesystem>

bool FSI::cell_is_in_fluid_domain(const DoFHandler<dim>::cell_iterator &cell) {
  return (cell->material_id() == fluid_domain_id);
}
bool FSI::cell_is_in_solid_domain(const DoFHandler<dim>::cell_iterator &cell) {
  return (cell->material_id() == solid_domain_id);
}

void FSI::make_grid() {
  // Build mesh on a serial triangulation, then create description for fully distributed mesh
  Triangulation<dim> mesh_serial;
  GridGenerator::subdivided_hyper_cube(mesh_serial, 8, -1, 1);

  for (const auto &cell : mesh_serial.active_cell_iterators())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && (face->center()[dim - 1] == 1))
        face->set_all_boundary_ids(1);

  for (const auto &cell : mesh_serial.active_cell_iterators())
    if (((std::fabs(cell->center()[0]) < 0.25) &&
         (cell->center()[dim - 1] > 0.5)) ||
        ((std::fabs(cell->center()[0]) >= 0.25) &&
         (cell->center()[dim - 1] > -0.5)))
      cell->set_material_id(fluid_domain_id);
    else
      cell->set_material_id(solid_domain_id);

  GridTools::partition_triangulation(mpi_size, mesh_serial);
  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      mesh_serial, MPI_COMM_WORLD);
  mesh.create_triangulation(construction_data);
}

void FSI::setup() {
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;
    // Mesh is generated in make_grid(); just report counts here
    pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
    {
      pcout << "Initializing the finite element space" << std::endl;
      // Setup FE space.
      const FE_Q<dim> fe_scalar_velocity(degree_velocity);
      const FE_Q<dim> fe_scalar_pressure(degree_pressure);
      const FE_Q<dim> fe_scalar_displacement(degree_displacement);
      // FE_Nothing describes a finite dimensional function space of functions that are constantly zero. It allows combining different FE systems.
      const FE_Nothing<dim> fe_nothing;
      stokes_fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity, dim, fe_scalar_pressure, 1, fe_nothing, dim);
      elasticity_fe = std::make_unique<FESystem<dim>>(fe_nothing, dim, fe_nothing, 1, fe_scalar_displacement, dim);
      fe_collection.push_back(*stokes_fe);
      fe_collection.push_back(*elasticity_fe);
      pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree << std::endl;
      pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree << std::endl;
      pcout << "  Displacement degree:           = " << fe_scalar_displacement.degree << std::endl;
      pcout << "  DoFs per cell              = " << stokes_fe->dofs_per_cell + elasticity_fe->dofs_per_cell << std::endl;
      // Initialize quadrature formulas for numerical integration. The order (degree + 1) ensures exact integration of the weak form.
      stokes_quadrature = std::make_unique<QGauss<dim>>(stokes_fe->degree + 2);
      elasticity_quadrature = std::make_unique<QGauss<dim>>(elasticity_fe->degree + 2);
      pcout << "  Stokes quadrature points per cell = " << stokes_quadrature->size() << std::endl;
      pcout << "  Elasticity quadrature points per cell = " << elasticity_quadrature->size() << std::endl;
      common_face_quadrature = std::make_unique<QGauss<dim - 1>>(std::max(stokes_fe->degree + 2, elasticity_fe->degree + 2));
      pcout << "  Quadrature points per face = " << common_face_quadrature->size() << std::endl;
    }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.

  {
    pcout << "Initializing the DoF handler" << std::endl;
    dof_handler.reinit(mesh);

    // This can be wrapped in a function as done in the tutorial

    // Assign active FE index to each cell based on its domain
    // Index 0 for fluid (Stokes), index 1 for solid (elasticity)
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (cell_is_in_fluid_domain(cell)) {
        cell->set_active_fe_index(0);
      } else if (cell_is_in_solid_domain(cell)) {
        cell->set_active_fe_index(1);
      } else {
        Assert(false, ExcNotImplemented());
      }
    }

    dof_handler.distribute_dofs(fe_collection);

    // Reorder DoFs so that all velocity DoFs come first, then
    // all pressure DoFs, and then all displacement DoFs.
    std::vector<unsigned int> block_component(dim + 1 + dim, 0);

    // Assign block index 0 to velocity, 1 to pressure, and 2 to displacement.
    // This sorts fluid velocity in block 0, pressure in block 1, and
    // solid displacement in block 2.
    block_component[dim] = 1;
    for (unsigned int i = dim + 1; i < block_component.size(); ++i) {
      block_component[i] = 2;
    }
    DoFRenumbering::component_wise(dof_handler, block_component);

    // Set up constraints for boundary conditions and hanging nodes.
    {
      constraints.clear();
      // Handle hanging nodes for mesh refinement compatibility.
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      const FEValuesExtractors::Vector velocities(0);
      VectorTools::interpolate_boundary_values(
        dof_handler,
        1,
        InletVelocity(),
        constraints,
        fe_collection.component_mask(velocities));

      const FEValuesExtractors::Vector displacements(dim + 1);
      VectorTools::interpolate_boundary_values(
        dof_handler,
        0,
        Functions::ZeroFunction<dim>(dim + 1 + dim),
        constraints,
        fe_collection.component_mask(displacements));
    }

    // Constrain velocity DoFs at the fluid-solid interface to enforce
    // the continuity condition.
    {
      std::vector<types::global_dof_index> local_face_dof_indices(stokes_fe->n_dofs_per_face());
      for (const auto &cell : dof_handler.active_cell_iterators()) {
        if (cell_is_in_fluid_domain(cell)) {
          for (const auto face_no : cell->face_indices()) {
            if (cell->face(face_no)->at_boundary() == false) {
              bool face_is_on_interface = false;

              if ((cell->neighbor(face_no)->has_children() == false) &&
                  (cell_is_in_solid_domain(cell->neighbor(face_no)))) {
                face_is_on_interface = true;
              } else if (cell->neighbor(face_no)->has_children() == true) {
                for (unsigned int sf = 0; sf < cell->face(face_no)->n_children(); ++sf) {
                  if (cell_is_in_solid_domain(cell->neighbor_child_on_subface(face_no, sf))) {
                    face_is_on_interface = true;
                    break;
                  }
                }
              }

              if (face_is_on_interface) {
                cell->face(face_no)->get_dof_indices(local_face_dof_indices, 0);
                for (unsigned int i = 0; i < local_face_dof_indices.size(); ++i) {
                  if (stokes_fe->face_system_to_component_index(i).first < dim) {
                    constraints.add_line(local_face_dof_indices[i]);
                  }
                }
              }
            }
          }
        }
      }
    }

    constraints.close();

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    // In addition to the locally owned and locally relevant indices for the whole
    // system, we also need those for the individual velocity, pressure, and displacement blocks.
    std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];
    const unsigned int n_d = dofs_per_block[2];

    block_owned_dofs.resize(3);
    block_relevant_dofs.resize(3);
    block_owned_dofs[0] = locally_owned_dofs.get_view(0, n_u);
    block_owned_dofs[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
    block_owned_dofs[2] = locally_owned_dofs.get_view(n_u + n_p, n_u + n_p + n_d);
    block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
    block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);
    block_relevant_dofs[2] = locally_relevant_dofs.get_view(n_u + n_p, n_u + n_p + n_d);

    pcout << "  Number of DoFs: " << std::endl;
    pcout << "    velocity = " << n_u << std::endl;
    pcout << "    pressure = " << n_p << std::endl;
    pcout << "    displacement = " << n_d << std::endl;
    pcout << "    total    = " << n_u + n_p + n_d << std::endl;

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system.

    pcout << "Initializing the linear system" << std::endl;
    pcout << "Initializing the sparsity pattern" << std::endl;
    // Velocity DoFs interact with other velocity DoFs (the weak formulation has
    // terms involving u·v), and pressure DoFs interact with velocity DoFs
    // (there are terms involving p·v or u·q). However, pressure
    // DoFs do not interact with other pressure DoFs (there are no
    // p·q terms). Displacement DoFs interact with other displacement DoFs.
    // Build a table to store this information so that the sparsity pattern
    // can be constructed accordingly.
    //DynamicSparsityPattern sparsity(dof_handler.n_dofs(), dof_handler.n_dofs());

    Table<2, DoFTools::Coupling> coupling(dim + 1 + dim, dim + 1 + dim);
    Table<2, DoFTools::Coupling> face_coupling(dim + 1 + dim, dim + 1 + dim);
    const unsigned int n_components = dim + 1 + dim;

    for (unsigned int c = 0; c < n_components; ++c)
      for (unsigned int d = 0; d < n_components; ++d) {
        coupling[c][d] = DoFTools::always;
        face_coupling[c][d] = DoFTools::always;
      }
    std::vector<types::global_dof_index> dofs_per_block_global = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

    // We use the standard dealii object and not the trilinos one.
    // Thi is just an intermediate step since we can not pass 
    // a trilinos objet to make_flux_sparsity_pattern
    BlockDynamicSparsityPattern dsp(3, 3);
    for (unsigned int i = 0; i < 3; ++i) {
      for (unsigned int j = 0; j < 3; ++j) {
        dsp.block(i, j).reinit(dofs_per_block_global[i], dofs_per_block_global[j]);
      }
    }
    dsp.collect_sizes();

    // The matrix entries from the interface terms must be included in the sparsity
    // pattern, so we use make_flux_sparsity_pattern. This function is similar
    // to make_sparsity_pattern() but assumes the bilinear form contains terms
    // that integrate over faces between cells. This is exactly what we need since
    // interface terms couple degrees of freedom from two adjacent cells along
    // a face. This also includes entries from computing terms coupling DoFs
    // from both sides of all interfaces.
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, coupling, face_coupling);

    // Ensure all diagonal entries exist (needed for constrained rows)
    for (unsigned int b = 0; b < 3; ++b)
      for (types::global_dof_index i = 0; i < dsp.block(b, b).n_rows(); ++i)
        dsp.block(b, b).add(i, i);

    constraints.condense(dsp);

    // Re-add diagonals after condense to preserve constrained rows
    for (unsigned int b = 0; b < 3; ++b)
      for (types::global_dof_index i = 0; i < dsp.block(b, b).n_rows(); ++i)
        dsp.block(b, b).add(i, i);

    TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs,
                                                    block_relevant_dofs,
                                                    block_owned_dofs,
                                                    MPI_COMM_WORLD);
    for (unsigned int i = 0; i < 3; ++i) {
      for (unsigned int j = 0; j < 3; ++j) {
        sparsity.block(i, j).reinit(block_owned_dofs[i],
                                    block_owned_dofs[j],
                                    dsp.block(i, j),
                                    MPI_COMM_WORLD,
                                    false);
      }
    }
    sparsity.collect_sizes();
    sparsity.compress();

    // Build a sparsity pattern for the pressure mass matrix.
    // what does this do precisely?
    Table<2, DoFTools::Coupling> pressure_coupling(dim + 1 + dim, dim + 1 + dim);
    for (unsigned int c = 0; c < dim + 1 + dim; ++c) {
      for (unsigned int d = 0; d < dim + 1 + dim; ++d) {
        pressure_coupling[c][d] = DoFTools::none;
      }
    }

    // Only pressure-pressure block is coupled
    pressure_coupling[dim][dim] = DoFTools::always;

    TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, pressure_coupling, sparsity_pressure_mass);
    sparsity_pressure_mass.compress();

    pressure_mass.reinit(sparsity_pressure_mass);
    pcout << "  Initializing the matrices" << std::endl;
    system_matrix.reinit(sparsity);

    mass_matrix.reinit(sparsity);
    mass_matrix = 0;

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
  }
}

void FSI::assemble_system() {
    // Initialize system_matrix and system_rhs to zero
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    const FEValuesExtractors::Vector displacements(dim + 1);

    system_matrix = 0;
    system_rhs = 0;
    // Initialize pressure mass matrix to zero
    pressure_mass = 0;
    // Assemble pressure mass matrix
    hp::QCollection<dim> q_collection_mass;
    q_collection_mass.push_back(*stokes_quadrature);
    q_collection_mass.push_back(*elasticity_quadrature);
    hp::FEValues<dim> hp_fe_values_mass(fe_collection,
                       q_collection_mass,
                       update_values | update_JxW_values);
    std::vector<types::global_dof_index> local_dof_indices_mass;
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (!cell_is_in_fluid_domain(cell))
        continue;
      hp_fe_values_mass.reinit(cell);
      const FEValues<dim> &fe_values = hp_fe_values_mass.get_present_fe_values();
      const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
      FullMatrix<double> local_mass_matrix(dofs_per_cell, dofs_per_cell);
      local_mass_matrix = 0;
      for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            local_mass_matrix(i, j) +=
              fe_values[pressure].value(i, q) *
              fe_values[pressure].value(j, q) *
              fe_values.JxW(q);
      local_dof_indices_mass.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices_mass);
      constraints.distribute_local_to_global(local_mass_matrix,
                          local_dof_indices_mass,
                          pressure_mass);
    }

    // Main system assembly
    hp::QCollection<dim> q_collection;
    q_collection.push_back(*stokes_quadrature);
    q_collection.push_back(*elasticity_quadrature);

    hp::FEValues<dim> hp_fe_values(fe_collection,
                                   q_collection,
                                   update_values | update_quadrature_points |
                                   update_JxW_values | update_gradients);

    FEFaceValues<dim> stokes_fe_face_values(*stokes_fe,
                                            *common_face_quadrature,
                                            update_JxW_values |
                                            update_gradients | update_values);
    FEFaceValues<dim> elasticity_fe_face_values(*elasticity_fe,
                                                *common_face_quadrature,
                                                update_normal_vectors |
                                                update_values);
    FESubfaceValues<dim> stokes_fe_subface_values(*stokes_fe,
                                                  *common_face_quadrature,
                                                  update_JxW_values |
                                                  update_gradients |
                                                  update_values);
    FESubfaceValues<dim> elasticity_fe_subface_values(*elasticity_fe,
                                                      *common_face_quadrature,
                                                      update_normal_vectors |
                                                      update_values);

    const unsigned int stokes_dofs_per_cell = stokes_fe->dofs_per_cell;
    const unsigned int elasticity_dofs_per_cell = elasticity_fe->dofs_per_cell;

    FullMatrix<double> local_matrix;
    FullMatrix<double> local_interface_matrix(elasticity_dofs_per_cell,
                                              stokes_dofs_per_cell);
    Vector<double> local_rhs;

    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<types::global_dof_index> neighbor_dof_indices(stokes_dofs_per_cell);

    // Storage for basis function values and derivatives.
    std::vector<Tensor<2, dim>> stokes_grad_phi_u(stokes_dofs_per_cell);
    std::vector<double> stokes_div_phi_u(stokes_dofs_per_cell);
    std::vector<double> stokes_phi_p(stokes_dofs_per_cell);

    std::vector<Tensor<2, dim>> elasticity_grad_phi(elasticity_dofs_per_cell);
    std::vector<double>         elasticity_div_phi(elasticity_dofs_per_cell);
    std::vector<Tensor<1, dim>> elasticity_phi(elasticity_dofs_per_cell);

    // Main loop over all cells
    for (const auto &cell : dof_handler.active_cell_iterators()) {
        /**
         * When looping over cells, we must check which domain each cell belongs to.
         * hp_fe_values checks the active_fe_index of the current cell:
         * index 0 uses the Stokes FE and quadrature rule,
         * index 1 uses the Elasticity FE and quadrature rule.
         * local_matrix.reinit() and local_rhs.reinit() resize the local system.
         * This is necessary since the number of DoFs may change across domains.
         */
        hp_fe_values.reinit(cell);
        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
        const unsigned int local_dofs = cell->get_fe().n_dofs_per_cell();
        local_matrix.reinit(local_dofs, local_dofs);
        local_rhs.reinit(local_dofs);

        /**
         * We use an if clause since on each cell, one set of variables
         * (either velocities and pressure, or displacements)
         * are always zero.
         */
        if (cell_is_in_fluid_domain(cell)) {
            // Verify that the cell's DoF count matches Stokes expectations.
            // This sanity check ensures consistency between the FE and the active cell.
            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
            Assert(dofs_per_cell == stokes_dofs_per_cell, ExcInternalError());

            const unsigned int n_q = fe_values.n_quadrature_points;

            for (unsigned int q = 0; q < n_q; ++q) {
                /**
                 * Store basis function k evaluated at quadrature point q.
                 * This allows fast lookups when iterating over i and j.
                 */
                for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                    stokes_grad_phi_u[k] =
                        fe_values[velocities].gradient(k, q);
                    stokes_div_phi_u[k] =
                        fe_values[velocities].divergence(k, q);
                    stokes_phi_p[k] =
                        fe_values[pressure].value(k, q);
                }

                // Assemble Stokes weak form:
                // ν(∇u_i, ∇u_j) - (∇·u_i, p_j) - (p_i, ∇·u_j)
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        local_matrix(i, j) +=
                            (nu * scalar_product(stokes_grad_phi_u[i],
                                                 stokes_grad_phi_u[j])
                             - stokes_div_phi_u[i] * stokes_phi_p[j]
                             - stokes_phi_p[i] * stokes_div_phi_u[j])
                            * fe_values.JxW(q);
                    }
                }
            }
        } else {
            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
            Assert(dofs_per_cell == elasticity_dofs_per_cell, ExcInternalError());

            const unsigned int n_q = fe_values.n_quadrature_points;

            for (unsigned int q = 0; q < n_q; ++q) {
                /**
                 * Store basis function k evaluated at quadrature point q.
                 * This allows fast lookups when iterating over i and j.
                 */
                for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                    elasticity_grad_phi[k] =
                        fe_values[displacements].gradient(k, q);
                    elasticity_div_phi[k] =
                        fe_values[displacements].divergence(k, q);
                }

                // Assemble linear elasticity weak form:
                // λ(∇·d_i, ∇·d_j) + μ(∇d_i, ∇d_j) + μ(∇d_i, (∇d_j)^T)
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        local_matrix(i, j) +=
                            (lambda * elasticity_div_phi[i] * elasticity_div_phi[j]
                             + mu * scalar_product(elasticity_grad_phi[i],
                                                  elasticity_grad_phi[j])
                             + mu * scalar_product(elasticity_grad_phi[i],
                                                  transpose(elasticity_grad_phi[j])))
                            * fe_values.JxW(q);
                    }
                }
            }
        }
        /**
         * Copy contributions from cell integrals into the global matrix.
         */
        local_dof_indices.resize(local_dofs);
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix,
                                              local_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
        /**
         * Assemble the face terms along the interface between the two subdomains.
         * To avoid assembling the interface twice (since looping over all faces
         * would encounter it twice), we arbitrarily evaluate interface terms
         * when the current cell is in the solid subdomain. If the neighbor is
         * a fluid cell, we are at the interface. We handle three cases
         * due to mesh refinement.
         */
        if (cell_is_in_solid_domain(cell)) {
            for (const auto f : cell->face_indices()) {
                if (cell->face(f)->at_boundary() == false) {
                    /**
                     * Case 1: The neighbor is at the same level, has no children,
                     * and is a fluid cell. The two cells share a boundary that is
                     * part of the interface where we integrate interface terms.
                     */
                    if ((cell->neighbor(f)->level() == cell->level()) &&
                        (cell->neighbor(f)->has_children() == false) &&
                        cell_is_in_fluid_domain(cell->neighbor(f))) {
                        elasticity_fe_face_values.reinit(cell, f);
                        stokes_fe_face_values.reinit(cell->neighbor(f),
                                                    cell->neighbor_of_neighbor(f));

                        assemble_interface_term(elasticity_fe_face_values,
                                               stokes_fe_face_values,
                                               elasticity_phi,
                                               stokes_grad_phi_u,
                                               stokes_phi_p,
                                               local_interface_matrix);

                        cell->neighbor(f)->get_dof_indices(neighbor_dof_indices);
                        constraints.distribute_local_to_global(
                            local_interface_matrix,
                            local_dof_indices,
                            neighbor_dof_indices,
                            system_matrix);
                    }
                    /**
                     * Case 2: The neighbor has further children.
                     * We loop over all children of the neighbor to check if they
                     * are part of the fluid subdomain.
                     */
                    else if ((cell->neighbor(f)->level() == cell->level()) &&
                             (cell->neighbor(f)->has_children() == true)) {
                        for (unsigned int subface = 0;
                             subface < cell->face(f)->n_children();
                             ++subface) {
                            if (cell_is_in_fluid_domain(
                                    cell->neighbor_child_on_subface(f, subface))) {
                                elasticity_fe_subface_values.reinit(cell, f, subface);
                                stokes_fe_face_values.reinit(
                                    cell->neighbor_child_on_subface(f, subface),
                                    cell->neighbor_of_neighbor(f));

                                assemble_interface_term(elasticity_fe_subface_values,
                                                       stokes_fe_face_values,
                                                       elasticity_phi,
                                                       stokes_grad_phi_u,
                                                       stokes_phi_p,
                                                       local_interface_matrix);

                                cell->neighbor_child_on_subface(f, subface)
                                    ->get_dof_indices(neighbor_dof_indices);
                                constraints.distribute_local_to_global(
                                    local_interface_matrix,
                                    local_dof_indices,
                                    neighbor_dof_indices,
                                    system_matrix);
                            }
                        }
                    }
                    /**
                     * Case 3: The neighbor is coarser.
                     */
                    else if (cell->neighbor_is_coarser(f) &&
                             cell_is_in_fluid_domain(cell->neighbor(f))) {
                        elasticity_fe_face_values.reinit(cell, f);
                        stokes_fe_subface_values.reinit(
                            cell->neighbor(f),
                            cell->neighbor_of_coarser_neighbor(f).first,
                            cell->neighbor_of_coarser_neighbor(f).second);

                        assemble_interface_term(elasticity_fe_face_values,
                                               stokes_fe_subface_values,
                                               elasticity_phi,
                                               stokes_grad_phi_u,
                                               stokes_phi_p,
                                               local_interface_matrix);

                        cell->neighbor(f)->get_dof_indices(neighbor_dof_indices);
                        constraints.distribute_local_to_global(
                            local_interface_matrix,
                            local_dof_indices,
                            neighbor_dof_indices,
                            system_matrix);
                    }
                }
            }
        }
    }
}

// Assemble the coupling terms at the fluid-solid interface.
// This enforces the kinematic coupling condition: fluid stress acting on solid.
// Interface term: ∫_Γ (2ν∇u·n - p·n) · d dΓ
void FSI::assemble_interface_term(
  const FEFaceValuesBase<dim> &         elasticity_fe_face_values,
  const FEFaceValuesBase<dim> &         stokes_fe_face_values,
  std::vector<Tensor<1, dim>> &         elasticity_phi,
  std::vector<Tensor<2, dim>> &stokes_grad_phi_u,
  std::vector<double> &                 stokes_phi_p,
  FullMatrix<double> &                  local_interface_matrix) const {
    Assert(stokes_fe_face_values.n_quadrature_points ==
           elasticity_fe_face_values.n_quadrature_points,
           ExcInternalError());
    const unsigned int n_face_quadrature_points =
        elasticity_fe_face_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    const FEValuesExtractors::Vector displacements(dim + 1);

    local_interface_matrix = 0;
    for (unsigned int q = 0; q < n_face_quadrature_points; ++q) {
        const Tensor<1, dim> normal_vector =
            elasticity_fe_face_values.normal_vector(q);

        for (unsigned int k = 0; k < stokes_fe_face_values.dofs_per_cell; ++k) {
            stokes_grad_phi_u[k] =
                stokes_fe_face_values[velocities].gradient(k, q);
            stokes_phi_p[k] = stokes_fe_face_values[pressure].value(k, q);
        }
        for (unsigned int k = 0; k < elasticity_fe_face_values.dofs_per_cell; ++k)
            elasticity_phi[k] =
                elasticity_fe_face_values[displacements].value(k, q);

        // Compute interface coupling: -(fluid stress · normal) · displacement test function
        // Term: -∫_Γ (2ν ε(u)·n - p n) · d_i dΓ
        for (unsigned int i = 0; i < elasticity_fe_face_values.dofs_per_cell; ++i)
            for (unsigned int j = 0; j < stokes_fe_face_values.dofs_per_cell; ++j) {
                // Compute symmetric part of the gradient: ε(u) = 0.5 * (∇u + ∇u^T)
                const Tensor<2, dim> sym_grad_u =
                    0.5 * (stokes_grad_phi_u[j] + transpose(stokes_grad_phi_u[j]));

                local_interface_matrix(i, j) +=
                    -((2 * nu * (sym_grad_u * normal_vector) -
                       stokes_phi_p[j] * normal_vector) *
                      elasticity_phi[i]) * stokes_fe_face_values.JxW(q);
            }
    }
}

void FSI::solve() {

    SolverControl solver_control(2000, 1e-6 * system_rhs.l2_norm());
    SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

    // Use the FSIPreconditioner defined in the header
    pcout << "Solving the linear system" << std::endl;
    solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
    pcout << "  " << solver_control.last_step() << " GMRES iterations"
        << std::endl;

    solution = solution_owned;
    constraints.distribute(solution);
}

void FSI::output(const unsigned int refinement_cycle) {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
    for (unsigned int d = 0; d < dim; ++d)
        solution_names.emplace_back("displacement");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
    for (unsigned int d = 0; d < dim; ++d)
        data_component_interpretation.push_back(
            DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    std::ofstream output(
        "solution-" + Utilities::int_to_string(refinement_cycle, 2) + ".vtk");
    data_out.write_vtk(output);
}

void FSI::refine_mesh() {
  // Allocate vectors to store the error estimation for each cell
  Vector<float> stokes_estimated_error_per_cell(mesh.n_active_cells());
  Vector<float> elasticity_estimated_error_per_cell(mesh.n_active_cells());

  // Create a container with the two quadrature rules for the error estimator
  const QGauss<dim - 1> stokes_face_quadrature(degree_velocity + 2);
  const QGauss<dim - 1> elasticity_face_quadrature(degree_displacement + 2);

  hp::QCollection<dim - 1> face_q_collection;
  face_q_collection.push_back(stokes_face_quadrature);
  face_q_collection.push_back(elasticity_face_quadrature);

  // Use KellyErrorEstimator to compute error indicators
  // for velocity on every cell
  const FEValuesExtractors::Vector velocities(0);
  KellyErrorEstimator<dim>::estimate(
    dof_handler,
    face_q_collection,
    std::map<types::boundary_id, const Function<dim> *>(),
    solution,
    stokes_estimated_error_per_cell,
    fe_collection.component_mask(velocities));

  const FEValuesExtractors::Vector displacements(dim + 1);
  KellyErrorEstimator<dim>::estimate(
    dof_handler,
    face_q_collection,
    std::map<types::boundary_id, const Function<dim> *>(),
    solution,
    elasticity_estimated_error_per_cell,
    fe_collection.component_mask(displacements));

  // Combine the two error estimates with experimentally determined weights
  stokes_estimated_error_per_cell *=
    4.0f / stokes_estimated_error_per_cell.l2_norm();
  elasticity_estimated_error_per_cell *=
    1.0f / elasticity_estimated_error_per_cell.l2_norm();

  Vector<float> estimated_error_per_cell(mesh.n_active_cells());
  estimated_error_per_cell += stokes_estimated_error_per_cell;
  estimated_error_per_cell += elasticity_estimated_error_per_cell;

  // Set error to zero for cells at the interface to prevent artificially
  // large error indicators on both sides of the interface between subdomains.
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

    for (const auto f : cell->face_indices()) {
      if (cell_is_in_solid_domain(cell)) {
        if ((cell->at_boundary(f) == false) &&
          (((cell->neighbor(f)->level() == cell->level()) &&
            (cell->neighbor(f)->has_children() == false) &&
            cell_is_in_fluid_domain(cell->neighbor(f))) ||
           ((cell->neighbor(f)->level() == cell->level()) &&
            (cell->neighbor(f)->has_children() == true) &&
            (cell_is_in_fluid_domain(
              cell->neighbor_child_on_subface(f, 0)))) ||
           (cell->neighbor_is_coarser(f) &&
            cell_is_in_fluid_domain(cell->neighbor(f))))) {
          estimated_error_per_cell(cell->active_cell_index()) = 0;
        }
      } else {
        if ((cell->at_boundary(f) == false) &&
          (((cell->neighbor(f)->level() == cell->level()) &&
            (cell->neighbor(f)->has_children() == false) &&
            cell_is_in_solid_domain(cell->neighbor(f))) ||
           ((cell->neighbor(f)->level() == cell->level()) &&
            (cell->neighbor(f)->has_children() == true) &&
            (cell_is_in_solid_domain(
              cell->neighbor_child_on_subface(f, 0)))) ||
           (cell->neighbor_is_coarser(f) &&
            cell_is_in_solid_domain(cell->neighbor(f))))) {
          estimated_error_per_cell(cell->active_cell_index()) = 0;
        }
      }
    }
  }

  // Mark cells for refinement (30%) and coarsening (0%) based on error estimates
  GridRefinement::refine_and_coarsen_fixed_number(mesh,
                          estimated_error_per_cell,
                          0.3,
                          0.0);
  // Execute the actual refinement
  mesh.execute_coarsening_and_refinement();
}


void FSI::run() {
    // Create the mesh.
    std::cout << "Creating the mesh" << std::endl;
    make_grid();
    std::cout << "  Number of elements = " << mesh.n_global_active_cells()
              << std::endl;

    for (unsigned int refinement_cycle = 0; refinement_cycle < 10 - 2 * dim;
         ++refinement_cycle) {
        std::cout << "Refinement cycle " << refinement_cycle << std::endl;

        if (refinement_cycle > 0)
            refine_mesh();

        setup();

        std::cout << "   Assembling..." << std::endl;
        assemble_system();

        std::cout << "   Solving..." << std::endl;
        solve();

        std::cout << "   Writing output..." << std::endl;
        output(refinement_cycle);

        std::cout << std::endl;
    }
}