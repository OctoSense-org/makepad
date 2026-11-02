use makepad_sdf_engine::makepad_csg_sdf::mesh_field::{MeshData, MeshField};

// The three csg_sdf mesh_field fixtures, with the original dimensions and winding.
pub fn cube(open: bool) -> MeshData {
    let positions = (0..8).map(|i| [if i & 1 == 0 { -1.0 } else { 1.0 }, if i & 2 == 0 { -1.0 } else { 1.0 }, if i & 4 == 0 { -1.0 } else { 1.0 }]).collect();
    let mut triangles = Vec::new();
    for (i, q) in [[0,2,3,1], [4,5,7,6], [0,1,5,4], [2,6,7,3], [0,4,6,2], [1,3,7,5]].iter().enumerate() {
        if open && i == 3 { continue; }
        triangles.extend([[q[0],q[1],q[2]], [q[0],q[2],q[3]]]);
    }
    MeshData { positions, triangles }
}

pub fn corpus() -> Vec<(&'static str, MeshField)> {
    let mut overlap = cube(false);
    let n = overlap.positions.len() as u32;
    overlap.positions.extend(overlap.positions.clone().iter().map(|p| [p[0] + 1.0, p[1], p[2]]));
    overlap.triangles.extend(overlap.triangles.clone().iter().map(|t| [t[0] + n, t[1] + n, t[2] + n]));
    vec![("cube", MeshField::new(cube(false))), ("cup", MeshField::new(cube(true))), ("overlap", MeshField::new(overlap))]
}
