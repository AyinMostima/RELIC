"""
RHESSys-style Templates for Hierarchical Scaling

Provides pre-configured templates mimicking RHESSys patch->hillslope->basin
hierarchy for testing and validation.

Templates:
    - small_basin: 100 patches -> 10 hillslopes -> 1 basin
    - test_hillslope: 10 patches -> 1 hillslope
    - complex_basin: Variable patch distribution with flow routing

Mathematical Expression:
    RHESSys aggregation:
    hillslope.variable = Sigma(patch.variable * patch.area) / hillslope.area
    basin.variable = Sigma(hillslope.variable * hillslope.area) / basin.area

References:
    - Tague, C.L. & Band, L.E. (2004). RHESSys: Regional Hydro-Ecologic Simulation System
    - Band, L.E. et al. (1993). Forest ecosystem processes at the watershed scale
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
from relic.core import device_policy

def get_rhessys_template(
    template_name: str = "small_basin",
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Get pre-configured RHESSys-style hierarchy template.
    
    Parameters:
        template_name : str
            Name of template to retrieve
        device : Optional[torch.device]
            Target device for tensors
            
    Returns:
        Dict[str, Any]: Topology configuration
    """
    device = device or (
        device_policy.resolve_device(default="auto")
    )
    
    if template_name == "small_basin":
        return _create_small_basin_template(device)
    elif template_name == "test_hillslope":
        return _create_test_hillslope_template(device)
    elif template_name == "complex_basin":
        return _create_complex_basin_template(device)
    else:
        raise ValueError(f"Unknown template: {template_name}")


def _create_small_basin_template(device: torch.device) -> Dict[str, Any]:
    """
    Create small basin template.
    
    Structure:
        - 100 patches (10x10 grid)
        - 10 hillslopes (10 patches each)
        - 1 basin
    """
    n_patches = 100
    n_hillslopes = 10
    patches_per_hillslope = 10
    
    # Create patch areas (slightly variable for realism)
    np.random.seed(42)  # For reproducibility
    patch_areas = 900 + 200 * np.random.randn(n_patches)  # ~1000 m^2 +/- variation
    patch_areas = np.maximum(patch_areas, 100)  # Ensure positive
    
    # Create hillslope areas (exact sum of constituent patches)
    hillslope_areas = np.zeros(n_hillslopes)
    for h in range(n_hillslopes):
        start_idx = h * patches_per_hillslope
        end_idx = start_idx + patches_per_hillslope
        # Ensure exact numerical consistency
        constituent_patches = patch_areas[start_idx:end_idx]
        hillslope_areas[h] = float(np.sum(constituent_patches))  # Force exact sum
    
    # Create parent-child mappings
    parent_map = {}
    children_map = {h: [] for h in range(n_hillslopes)}
    
    for p in range(n_patches):
        hillslope = p // patches_per_hillslope
        parent_map[p] = hillslope
        children_map[hillslope].append(p)
    
    # Create flow routing (optional, for water routing)
    flow_table = _create_simple_flow_table(n_patches, n_hillslopes)
    
    return {
        'n_source_units': n_patches,
        'n_target_units': n_hillslopes,
        'source_areas': patch_areas.tolist(),
        'target_areas': hillslope_areas.tolist(),
        'parent_map': parent_map,
        'children_map': children_map,
        'flow_table': flow_table,
        'metadata': {
            'template': 'small_basin',
            'description': '100 patches -> 10 hillslopes',
            'patch_grid': (10, 10),
            'coordinate_system': 'projected',
            'units': 'meters'
        }
    }


def _create_test_hillslope_template(device: torch.device) -> Dict[str, Any]:
    """
    Create single hillslope template for testing.
    
    Structure:
        - 10 patches
        - 1 hillslope
    """
    n_patches = 10
    n_hillslopes = 1
    
    # Create patch areas with elevation gradient
    patch_areas = np.array([
        1000, 950, 900, 850, 800,  # Upper slope
        750, 700, 650, 600, 550     # Lower slope
    ])
    
    # Single hillslope area
    hillslope_areas = np.array([patch_areas.sum()])
    
    # Mappings
    parent_map = {p: 0 for p in range(n_patches)}
    children_map = {0: list(range(n_patches))}
    
    # Flow routing (downslope)
    flow_table = {
        'from_patch': list(range(9)),
        'to_patch': list(range(1, 10)),
        'flow_fraction': [1.0] * 9  # All flow goes downslope
    }
    
    return {
        'n_source_units': n_patches,
        'n_target_units': n_hillslopes,
        'source_areas': patch_areas.tolist(),
        'target_areas': hillslope_areas.tolist(),
        'parent_map': parent_map,
        'children_map': children_map,
        'flow_table': flow_table,
        'metadata': {
            'template': 'test_hillslope',
            'description': '10 patches -> 1 hillslope (with elevation gradient)',
            'elevation_gradient': 'linear',
            'flow_direction': 'downslope'
        }
    }


def _create_complex_basin_template(device: torch.device) -> Dict[str, Any]:
    """
    Create complex basin with variable patch distribution.
    
    Structure:
        - 237 patches (irregular distribution)
        - 15 hillslopes (variable sizes)
        - 3 sub-basins
        - 1 basin
    """
    # Define hillslope patch counts (variable)
    patches_per_hillslope = [
        20, 18, 15, 12, 10,  # Sub-basin 1
        25, 22, 20, 18, 15,  # Sub-basin 2
        17, 14, 11, 8, 7     # Sub-basin 3
    ]
    n_hillslopes = len(patches_per_hillslope)
    n_patches = sum(patches_per_hillslope)
    
    # Create patch areas with spatial correlation
    np.random.seed(42)
    patch_areas = []
    
    for h_idx, n_p in enumerate(patches_per_hillslope):
        # Base area varies by hillslope
        base_area = 800 + h_idx * 50
        # Add spatially correlated noise
        areas = base_area + 100 * np.random.randn(n_p)
        areas = np.maximum(areas, 100)  # Ensure positive
        patch_areas.extend(areas)
    
    patch_areas = np.array(patch_areas)
    
    # Calculate hillslope areas
    hillslope_areas = []
    patch_idx = 0
    
    for n_p in patches_per_hillslope:
        h_area = patch_areas[patch_idx:patch_idx + n_p].sum()
        hillslope_areas.append(h_area)
        patch_idx += n_p
    
    hillslope_areas = np.array(hillslope_areas)
    
    # Create mappings
    parent_map = {}
    children_map = {h: [] for h in range(n_hillslopes)}
    
    patch_idx = 0
    for h_idx, n_p in enumerate(patches_per_hillslope):
        for p in range(n_p):
            global_p = patch_idx + p
            parent_map[global_p] = h_idx
            children_map[h_idx].append(global_p)
        patch_idx += n_p
    
    # Create complex flow routing
    flow_table = _create_complex_flow_table(
        n_patches, n_hillslopes, patches_per_hillslope
    )
    
    # Add sub-basin information
    sub_basin_map = {
        0: list(range(5)),      # Sub-basin 1
        1: list(range(5, 10)),  # Sub-basin 2
        2: list(range(10, 15))  # Sub-basin 3
    }
    
    return {
        'n_source_units': n_patches,
        'n_target_units': n_hillslopes,
        'source_areas': patch_areas.tolist(),
        'target_areas': hillslope_areas.tolist(),
        'parent_map': parent_map,
        'children_map': children_map,
        'flow_table': flow_table,
        'sub_basin_map': sub_basin_map,
        'metadata': {
            'template': 'complex_basin',
            'description': f'{n_patches} patches -> {n_hillslopes} hillslopes -> 3 sub-basins',
            'patches_per_hillslope': patches_per_hillslope,
            'has_flow_routing': True,
            'has_sub_basins': True
        }
    }


def _create_simple_flow_table(
    n_patches: int,
    n_hillslopes: int
) -> Dict[str, List]:
    """
    Create simple downslope flow routing table.
    
    Flow goes from higher to lower patches within hillslopes.
    """
    flow_table = {
        'from_patch': [],
        'to_patch': [],
        'flow_fraction': []
    }
    
    patches_per_hillslope = n_patches // n_hillslopes
    
    for h in range(n_hillslopes):
        start = h * patches_per_hillslope
        end = start + patches_per_hillslope
        
        # Flow within hillslope (downslope)
        for p in range(start, end - 1):
            flow_table['from_patch'].append(p)
            flow_table['to_patch'].append(p + 1)
            flow_table['flow_fraction'].append(1.0)
    
    return flow_table


def _create_complex_flow_table(
    n_patches: int,
    n_hillslopes: int,
    patches_per_hillslope: List[int]
) -> Dict[str, List]:
    """
    Create complex flow routing with divergence and convergence.
    
    Includes:
        - Divergent flow at ridges
        - Convergent flow in valleys
        - Cross-hillslope flow
    """
    flow_table = {
        'from_patch': [],
        'to_patch': [],
        'flow_fraction': [],
        'flow_type': []  # 'surface', 'subsurface', 'channel'
    }
    
    patch_idx = 0
    
    for h_idx, n_p in enumerate(patches_per_hillslope):
        for p in range(n_p - 1):
            global_p = patch_idx + p
            
            # Primary downslope flow
            flow_table['from_patch'].append(global_p)
            flow_table['to_patch'].append(global_p + 1)
            flow_table['flow_fraction'].append(0.7)  # 70% downslope
            flow_table['flow_type'].append('surface')
            
            # Lateral flow (if not at boundary)
            if h_idx < n_hillslopes - 1 and p < patches_per_hillslope[h_idx + 1]:
                # Flow to adjacent hillslope
                adj_patch = sum(patches_per_hillslope[:h_idx + 1]) + p
                if adj_patch < n_patches:
                    flow_table['from_patch'].append(global_p)
                    flow_table['to_patch'].append(adj_patch)
                    flow_table['flow_fraction'].append(0.2)  # 20% lateral
                    flow_table['flow_type'].append('subsurface')
            
            # Convergent flow in lower patches
            if p > n_p // 2:  # Lower half of hillslope
                # Additional convergent flow
                if p > 0:
                    flow_table['from_patch'].append(global_p - 1)
                    flow_table['to_patch'].append(global_p + 1)
                    flow_table['flow_fraction'].append(0.1)  # 10% bypass
                    flow_table['flow_type'].append('channel')
        
        patch_idx += n_p
    
    return flow_table


def create_validation_data(
    template_name: str = "small_basin",
    variable_type: str = "extensive",
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Create validation data for testing aggregation.
    
    Parameters:
        template_name : str
            Template to use
        variable_type : str
            Type of variable ('extensive', 'intensive', 'categorical')
        device : Optional[torch.device]
            Device for tensors
            
    Returns:
        Dict[str, Tensor]: Test data for aggregation
    """
    device = device or (
        device_policy.resolve_device(default="auto")
    )
    template = get_rhessys_template(template_name, device)
    
    n_patches = template['n_source_units']
    
    # Create test data based on variable type
    if variable_type == "extensive":
        # Carbon mass fluxes in kg C m-2 s-1: GPP is gross_primary_productivity (the catalog name
        # gpp is the molar flux in umol m-2 s-1).
        data = {
            'gross_primary_productivity': torch.randn(n_patches, device=device).abs() * 1e-6,
            'respiration': torch.randn(n_patches, device=device).abs() * 5e-7,
            'nee': torch.randn(n_patches, device=device) * 1e-6  # Can be negative
        }
    elif variable_type == "intensive":
        # State variable (e.g., temperature K, LAI m2/m2)
        data = {
            'temperature': 273.15 + torch.randn(n_patches, device=device) * 10,  # ~273K +/- 10
            'lai': torch.randn(n_patches, device=device).abs() * 5,  # 0-5 m2/m2
            'soil_moisture': torch.rand(n_patches, device=device)  # 0-1 fraction
        }
    elif variable_type == "categorical":
        # Categorical variable (e.g., vegetation type)
        data = {
            'vegetation_type': torch.randint(0, 5, (n_patches,), device=device).float(),
            'soil_type': torch.randint(0, 3, (n_patches,), device=device).float()
        }
    else:
        raise ValueError(f"Unknown variable type: {variable_type}")
    
    return data


def validate_aggregation_conservation(
    source_data: torch.Tensor,
    aggregated_data: torch.Tensor,
    template: Dict[str, Any],
    tolerance: float = 1e-9
) -> Dict[str, Any]:
    """
    Validate conservation in aggregation.
    
    Parameters:
        source_data : Tensor
            Original patch-level data
        aggregated_data : Tensor
            Aggregated hillslope-level data
        template : Dict[str, Any]
            Template configuration
        tolerance : float
            Conservation tolerance
            
    Returns:
        Dict[str, Any]: Validation results
    """
    # Get areas
    source_areas = torch.tensor(template['source_areas'])
    target_areas = torch.tensor(template['target_areas'])
    
    # Compute totals
    source_total = (source_data * source_areas).sum()
    target_total = (aggregated_data * target_areas).sum()
    
    # Compute error
    absolute_error = abs(target_total - source_total)
    if source_total != 0:
        relative_error = absolute_error / abs(source_total)
    else:
        relative_error = absolute_error
    
    # Check conservation
    is_conserved = relative_error <= tolerance
    
    return {
        'source_total': float(source_total),
        'target_total': float(target_total),
        'absolute_error': float(absolute_error),
        'relative_error': float(relative_error),
        'tolerance': tolerance,
        'is_conserved': is_conserved,
        'message': 'Conservation satisfied' if is_conserved else f'Conservation violated: {relative_error:.2e} > {tolerance}'
    }
