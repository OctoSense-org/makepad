//! A shape lowered to the engine's program: a flat stream of f32 RGBA texels
//! that the bake shader and the CPU twin ([`crate::twin`]) both interpret, op
//! for op, so their results agree bit for bit.
//!
//! The stream is the node tree walked depth first:
//! - a leaf pushes its distance (and the material slot in force) on the value
//!   stack;
//! - a Combine pushes its first operand, then for each next one its program
//!   followed by a FOLD that combines the top two values (the slot rules of
//!   `csg_sdf::distance_slot`);
//! - point ops bracket their subtree: XFORM_PUSH moves the point into the
//!   child's frame and XFORM_POP restores it and scales the distance back;
//!   DEFORM_PUSH / DEFORM_POP warp it and divide by the warp's Lipschitz bound;
//! - Shell and Round act on the top value; MAT_PUSH / MAT_POP set the slot of
//!   the leaves inside (the innermost wins); a Material with a `within`
//!   region runs the region's program first and MAT_WITHIN pops its value:
//!   the slot applies where it is <= 0, elsewhere the inherited one stays;
//! - Instance becomes a sharp union of its copies (each one an XFORM bracket
//!   around the same subprogram); Bone and Id leave the field as it is;
//! - a Texture pushes its source's values at the four corners of its gradient's
//!   tetrahedron (XFORM brackets of a translation; only for its masks or the
//!   triplanar blend), the source's value, then its pattern kernel's value per
//!   projection (KERNEL ops in XFORM brackets that turn the point into the
//!   kernel's frame and units, no SCALE: a pattern is no distance); TEX
//!   combines them into the textured value. A single-projection texture's
//!   corners follow a GRAD op and its pattern a PAT op, which a march skips in a
//!   brick where the bake found the texture's masks constant ([`KNOWN_TEXTURES`]).
//!
//! All parameters are converted to f32 here, once: what the GPU and the twin
//! compute from is the same bits. A program starts with one header texel
//! (op-stream end, value-stack depth, point-stack depth, 0). Payloads follow
//! the op stream and count toward `texels.len()` and buffer uploads.

use crate::kernel::KernelSlot;
use makepad_csg_sdf::{SdfDeform, SdfNode, SdfOp, SdfPlace, TexProject};

/// Op codes (the header texel's `x`).
pub mod op {
    pub const PRIM: f32 = 0.0;
    pub const XFORM_PUSH: f32 = 1.0;
    pub const XFORM_POP: f32 = 2.0;
    pub const DEFORM_PUSH: f32 = 3.0;
    pub const DEFORM_POP: f32 = 4.0;
    pub const FOLD: f32 = 5.0;
    pub const SHELL: f32 = 6.0;
    pub const ROUND: f32 = 7.0;
    pub const MAT_PUSH: f32 = 8.0;
    pub const MAT_POP: f32 = 9.0;
    pub const MAT_WITHIN: f32 = 10.0;
    /// A compiled Splash kernel: (op, slot in the volume's kernel table,
    /// parameter texels, parameter count), then its parameters, four per
    /// texel. It pushes its distance like a primitive. A gridded kernel's
    /// count is [`super::GRIDDED`] more, and its parameter 0 is the texel
    /// offset from this op of its grid's words (a payload; the shader's
    /// `kernel_at_fetch` adds the op's own texel).
    pub const KERNEL: f32 = 11.0;
    /// The top value times `y`: a kernel's distance brought to a 1-Lipschitz
    /// bound (the volume's estimate of its steepest gradient).
    pub const SCALE: f32 = 12.0;
    /// Exact mesh query: y is the immutable asset binding in the field table.
    pub const MESH: f32 = 14.0;
    /// y/z: program-relative payload offset and texel count, stored as u32 bits.
    pub const LINKS: f32 = 15.0;
    /// A texture (`csg_sdf::SdfTexture::combine`): 3 texels, (op, triplanar, gradient,
    /// depth), (1 / lipschitz, 1 / (4 step), its ordinal, 0), (top ramp lo hi, bottom ramp lo
    /// hi; an absent mask is the ramp 2..3, 1 everywhere). It pops the pattern's values (3 for
    /// triplanar, else 1), the source's value and, with the gradient, the source's values at
    /// the tetrahedron's corners below it, and pushes the textured value with the source's slot.
    pub const TEX: f32 = 16.0;
    /// (op, texels of the gradient's four corner brackets after it, the texture's ordinal, 0):
    /// where the evaluation knows the texture's masks, it skips the brackets and pushes four
    /// zeros in their place (the TEX takes the known masks instead of the gradient's).
    pub const GRAD: f32 = 17.0;
    /// (op, texels of the pattern's brackets after it, the texture's ordinal, its values):
    /// where the texture's masks are known 0, it skips them and pushes zeros in their place.
    pub const PAT: f32 = 18.0;
}

/// Added to a KERNEL op's parameter count when the kernel reads a grid.
pub const GRIDDED: usize = 32;

/// The textures (in lowering order) whose masks an evaluation can be told: the ordinal's bit
/// of a known-masks word (`compute.rs`'s `run_program`), its value 16 bits up.
pub const KNOWN_TEXTURES: usize = 16;

/// The texels of the op whose header is `h` (its parameters included).
pub fn op_len(h: [f32; 4]) -> usize {
    if h[0] == op::PRIM || h[0] == op::TEX {
        3
    } else if h[0] == op::XFORM_PUSH {
        4
    } else if h[0] == op::KERNEL {
        1 + h[2] as usize
    } else {
        1
    }
}

/// Primitive kinds (a PRIM's header `y`). A PRIM is 3 texels: header
/// (op, kind, a, b) then (c, d, e, f) and (g, h, i, j).
pub mod prim {
    /// a = radius
    pub const SPHERE: f32 = 0.0;
    /// a b c = half size less the rounding, d = rounding
    pub const BOX: f32 = 1.0;
    /// a = radius, b = half height
    pub const CYLINDER: f32 = 2.0;
    /// a = bottom radius, b = top radius, c = half height
    pub const CONE: f32 = 3.0;
    /// a = ring radius, b = tube radius
    pub const TORUS: f32 = 4.0;
    /// a b c = start, d e f = end - start, g = radius
    pub const CAPSULE: f32 = 5.0;
    /// a b c = radii
    pub const ELLIPSOID: f32 = 6.0;
    /// a = 2 pi / period, b = half wall
    pub const GYROID: f32 = 7.0;
}

/// Why a shape could not be lowered.
#[derive(Clone, Debug, PartialEq)]
pub enum LowerError {
    /// A kernel or mesh the volume's kernel table lacks.
    Unsupported(&'static str),
    Kernel(String),
}

/// A lowered program.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Program {
    /// The texels, header first.
    pub texels: Vec<[f32; 4]>,
    /// The deepest value stack and point stack it needs.
    pub value_depth: usize,
    pub point_depth: usize,
}

/// The value stack and point stack every interpreter (shader and twin)
/// provides. Lowering rejects a deeper program before allocating bricks.
pub const MAX_VALUE_DEPTH: usize = 16;
pub const MAX_POINT_DEPTH: usize = 8;

impl Program {
    /// The program with its result multiplied by `s` (a field's Lipschitz scale, at the root:
    /// the values its nodes combine are untouched).
    pub fn scaled(mut self, s: f32) -> Program {
        if s != 1.0 {
            let end = self.texels[0][0] as usize;
            let mut pc = 1;
            while pc < end {
                let h = self.texels[pc];
                if h[0] == op::LINKS {
                    self.texels[pc][1] = f32::from_bits(h[1].to_bits() + 1);
                }
                if h[0] == op::KERNEL && h[3] as usize >= GRIDDED {
                    self.texels[pc + 1][0] += 1.0;
                }
                pc += op_len(h);
            }
            self.texels.insert(end, [op::SCALE, s, 0.0, 0.0]);
            self.texels[0][0] = (end + 1) as f32;
        }
        self
    }
}

struct Lower<'a> {
    kernels: &'a [KernelSlot],
    /// Multiplies every kernel's distance (1 / its Lipschitz bound).
    kernel_scale: f32,
    out: Vec<[f32; 4]>,
    payloads: Vec<(usize, Vec<[f32; 4]>)>,
    /// Gridded KERNEL ops and their grids' words.
    grids: Vec<(usize, Vec<[f32; 4]>)>,
    vdepth: usize,
    vmax: usize,
    pdepth: usize,
    pmax: usize,
    /// Textures lowered so far (the next one's ordinal).
    textures: usize,
}

fn f(v: f64) -> f32 {
    v as f32
}

/// The world-to-local affine of a placement (its rows: 3 x (a b c t)) and its scale.
fn affine_into(r: [[f64; 3]; 3], at: [f64; 3], scale: f64) -> [[f32; 4]; 3] {
    // local = R^T (p - at) / scale
    std::array::from_fn(|i| {
        let a = [r[0][i] / scale, r[1][i] / scale, r[2][i] / scale];
        let t = -(a[0] * at[0] + a[1] * at[1] + a[2] * at[2]);
        [f(a[0]), f(a[1]), f(a[2]), f(t)]
    })
}

/// The rotation of `csg_sdf`'s Transform: about X, then Y, then Z (degrees).
fn rot_deg(deg: [f64; 3]) -> [[f64; 3]; 3] {
    let (sx, cx) = deg[0].to_radians().sin_cos();
    let (sy, cy) = deg[1].to_radians().sin_cos();
    let (sz, cz) = deg[2].to_radians().sin_cos();
    let rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
    let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
    let m = |a: [[f64; 3]; 3], b: [[f64; 3]; 3]| -> [[f64; 3]; 3] { std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum())) };
    m(rz, m(ry, rx))
}

/// The rotation matrix of a unit quaternion (x y z w).
fn rot_quat(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [x, y, z, w] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

impl Lower<'_> {
    fn push_value(&mut self) {
        self.vdepth += 1;
        self.vmax = self.vmax.max(self.vdepth);
    }

    fn prim(&mut self, kind: f32, p: [f32; 10]) {
        self.out.push([op::PRIM, kind, p[0], p[1]]);
        self.out.push([p[2], p[3], p[4], p[5]]);
        self.out.push([p[6], p[7], p[8], p[9]]);
        self.push_value();
    }

    /// A KERNEL op: the kernel in `slot` at the current point, its parameters inline.
    fn kernel_op(&mut self, slot: usize, params: &[f32]) {
        let texels = params.len().div_ceil(4);
        self.out.push([op::KERNEL, slot as f32, texels as f32, params.len() as f32]);
        for c in params.chunks(4) {
            let mut t = [0.0f32; 4];
            t[..c.len()].copy_from_slice(c);
            self.out.push(t);
        }
        self.push_value();
    }

    fn xform(&mut self, rows: [[f32; 4]; 3], scale: f64, body: impl FnOnce(&mut Self) -> Result<(), LowerError>) -> Result<(), LowerError> {
        self.out.push([op::XFORM_PUSH, f(scale), 0.0, 0.0]);
        self.out.extend(rows);
        self.pdepth += 1;
        self.pmax = self.pmax.max(self.pdepth);
        body(self)?;
        self.pdepth -= 1;
        self.out.push([op::XFORM_POP, f(scale), 0.0, 0.0]);
        Ok(())
    }

    fn place(&mut self, pl: &SdfPlace, of: &SdfNode) -> Result<(), LowerError> {
        self.xform(affine_into(rot_quat(pl.rotate), pl.at, pl.scale), pl.scale, |s| s.node(of))
    }

    fn node(&mut self, n: &SdfNode) -> Result<(), LowerError> {
        let z = 0.0f32;
        match n {
            SdfNode::Sphere { radius } => self.prim(prim::SPHERE, [f(*radius), z, z, z, z, z, z, z, z, z]),
            SdfNode::Box { size, radius } => self.prim(prim::BOX, [f(size[0] * 0.5 - radius), f(size[1] * 0.5 - radius), f(size[2] * 0.5 - radius), f(*radius), z, z, z, z, z, z]),
            SdfNode::Cylinder { radius, top_radius, height } => {
                if radius == top_radius {
                    self.prim(prim::CYLINDER, [f(*radius), f(height * 0.5), z, z, z, z, z, z, z, z])
                } else {
                    self.prim(prim::CONE, [f(*radius), f(*top_radius), f(height * 0.5), z, z, z, z, z, z, z])
                }
            }
            SdfNode::Torus { radius, tube } => self.prim(prim::TORUS, [f(*radius), f(*tube), z, z, z, z, z, z, z, z]),
            SdfNode::Capsule { a, b, radius } => {
                self.prim(prim::CAPSULE, [f(a[0]), f(a[1]), f(a[2]), f(b[0] - a[0]), f(b[1] - a[1]), f(b[2] - a[2]), f(*radius), z, z, z])
            }
            SdfNode::Ellipsoid { radii } => self.prim(prim::ELLIPSOID, [f(radii[0]), f(radii[1]), f(radii[2]), z, z, z, z, z, z, z]),
            SdfNode::Gyroid { period, wall } => self.prim(prim::GYROID, [f(std::f64::consts::TAU / period), f(wall * 0.5), z, z, z, z, z, z, z, z]),
            SdfNode::Combine { op: o, blend, of } => {
                self.node(&of[0])?;
                let code = match o {
                    SdfOp::Union => 0.0,
                    SdfOp::Difference => 1.0,
                    SdfOp::Intersection => 2.0,
                };
                for c in &of[1..] {
                    self.node(c)?;
                    self.out.push([op::FOLD, code, f(*blend), 0.0]);
                    self.vdepth -= 1;
                }
            }
            SdfNode::Shell { thickness, of } => {
                self.node(of)?;
                self.out.push([op::SHELL, f(*thickness), 0.0, 0.0]);
            }
            SdfNode::Round { radius, of } => {
                self.node(of)?;
                self.out.push([op::ROUND, f(*radius), 0.0, 0.0]);
            }
            SdfNode::Transform { at, rotate, scale, of } => self.xform(affine_into(rot_deg(*rotate), *at, *scale), *scale, |s| s.node(of))?,
            SdfNode::Material { slot, within, of } => {
                match within {
                    Some(w) => {
                        self.node(w)?;
                        self.out.push([op::MAT_WITHIN, *slot as f32, 0.0, 0.0]);
                        self.vdepth -= 1;
                    }
                    None => self.out.push([op::MAT_PUSH, *slot as f32, 0.0, 0.0]),
                }
                self.node(of)?;
                self.out.push([op::MAT_POP, 0.0, 0.0, 0.0]);
            }
            SdfNode::Instance { places, of } => {
                for (i, pl) in places.iter().enumerate() {
                    self.place(pl, of)?;
                    if i > 0 {
                        self.out.push([op::FOLD, 0.0, 0.0, 0.0]);
                        self.vdepth -= 1;
                    }
                }
            }
            SdfNode::Bone { of, .. } | SdfNode::Id { of, .. } => self.node(of)?,
            SdfNode::Links(l) => {
                let data = links_data(l);
                self.payloads.push((self.out.len(), data));
                self.out.push([op::LINKS, 0.0, 0.0, 0.0]);
                self.push_value();
            }
            // the capsules of its strokes (a brick's program holds only the ones near it)
            SdfNode::Sculpt { strokes, of } => {
                let all = ([f64::NEG_INFINITY; 3], [f64::INFINITY; 3]);
                match makepad_csg_sdf::bounds::sculpt_in(strokes, of, &all) {
                    Some(n) => self.node(&n)?,
                    None => return Err(LowerError::Unsupported("a sculpt with nothing left of its solid")),
                }
            }
            SdfNode::Texture { tex, of } => {
                let Some(slot) = self.kernels.iter().position(|s| s.matches(&tex.kernel)) else {
                    return Err(LowerError::Unsupported("a texture's kernel missing from the volume's kernel table"));
                };
                let grad = tex.needs_gradient();
                let triplanar = tex.project == TexProject::Triplanar;
                let ordinal = self.textures;
                self.textures += 1;
                // A single projection's masks read the gradient's Y alone: where they are
                // constant over a brick, a march is told and skips the corners (and, masked
                // off, the pattern). The triplanar blend needs the whole gradient.
                let skips = !triplanar && ordinal < KNOWN_TEXTURES;
                if grad {
                    let at = self.out.len();
                    if skips { self.out.push([op::GRAD, 0.0, ordinal as f32, 0.0]); }
                    for k in makepad_csg_sdf::texture::TETRA {
                        let t = k.map(|c| f(c * tex.step));
                        self.xform([[1.0, 0.0, 0.0, t[0]], [0.0, 1.0, 0.0, t[1]], [0.0, 0.0, 1.0, t[2]]], 1.0, |s| s.node(of))?;
                    }
                    if skips { self.out[at][1] = (self.out.len() - at - 1) as f32; }
                }
                self.node(of)?;
                let axes: &[u8] = match tex.project {
                    TexProject::Axis(a) => &[0, 1, 2][a.min(2) as usize..a.min(2) as usize + 1],
                    TexProject::Triplanar => &[0, 1, 2],
                };
                let params = tex.kernel.splash.as_ref().map(|s| s.params()).unwrap_or_default();
                let k = f(tex.kscale);
                let pat = self.out.len();
                if skips && grad { self.out.push([op::PAT, 0.0, ordinal as f32, axes.len() as f32]); }
                for &a in axes {
                    // the kernel's point: the swizzle that makes axis `a` its Y, in its units
                    let rows = match a {
                        0 => [[0.0, 0.0, k, 0.0], [k, 0.0, 0.0, 0.0], [0.0, k, 0.0, 0.0]],
                        2 => [[k, 0.0, 0.0, 0.0], [0.0, 0.0, k, 0.0], [0.0, k, 0.0, 0.0]],
                        _ => [[k, 0.0, 0.0, 0.0], [0.0, k, 0.0, 0.0], [0.0, 0.0, k, 0.0]],
                    };
                    self.xform(rows, 1.0, |s| {
                        s.kernel_op(slot, &params);
                        Ok(())
                    })?;
                }
                if skips && grad { self.out[pat][1] = (self.out.len() - pat - 1) as f32; }
                let none = (2.0, 3.0);
                let (top, bottom) = (tex.top.unwrap_or(none), tex.bottom.unwrap_or(none));
                self.out.push([op::TEX, if triplanar { 1.0 } else { 0.0 }, if grad { 1.0 } else { 0.0 }, f(tex.depth)]);
                self.out.push([f(1.0 / tex.lipschitz), f(0.25 / tex.step), ordinal as f32, 0.0]);
                self.out.push([f(top.0), f(top.1), f(bottom.0), f(bottom.1)]);
                self.vdepth -= axes.len() + if grad { 4 } else { 0 };
            }
            SdfNode::Deform { deform, lipschitz, of } => {
                let (kind, rate, axis) = match *deform {
                    SdfDeform::Twist { rate } => (0.0, rate, 0),
                    SdfDeform::Bend { rate } => (1.0, rate, 0),
                    SdfDeform::Taper { rate } => (2.0, rate, 0),
                    SdfDeform::Mirror { axis } => (3.0, 0.0, axis.min(2)),
                };
                self.out.push([op::DEFORM_PUSH, kind, f(rate), axis as f32]);
                self.pdepth += 1;
                self.pmax = self.pmax.max(self.pdepth);
                self.node(of)?;
                self.pdepth -= 1;
                self.out.push([op::DEFORM_POP, f(*lipschitz), 0.0, 0.0]);
            }
            SdfNode::Kernel(k) => {
                let Some(slot) = self.kernels.iter().position(|s| s.matches(k)) else {
                    return Err(LowerError::Unsupported("a kernel missing from the volume's kernel table"));
                };
                let params = k.splash.as_ref().map(|s| s.params()).unwrap_or_default();
                let at = self.out.len();
                self.kernel_op(slot, &params);
                if let Some(g) = k.splash.as_ref().and_then(|s| s.grid()) {
                    self.out[at][3] += GRIDDED as f32;
                    let mut words = g.words();
                    words.resize(words.len().div_ceil(4) * 4, 0.0);
                    self.grids.push((at, words.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()));
                }
                if self.kernel_scale != 1.0 {
                    self.out.push([op::SCALE, self.kernel_scale, 0.0, 0.0]);
                }
            }
            // A mesh leaf uses the same exact query in the bake and layer trace.
            SdfNode::Mesh(m) => {
                let Some(slot) = self.kernels.iter().position(|s| s.key == m.key() && s.is_mesh()) else {
                    return Err(LowerError::Unsupported("a mesh missing from the volume's kernel table"));
                };
                self.out.push([op::MESH, slot as f32, 0.0, 0.0]);
                self.push_value();
                if self.kernel_scale != 1.0 {
                    self.out.push([op::SCALE, self.kernel_scale, 0.0, 0.0]);
                }
            }
        }
        Ok(())
    }
}

/// Lowers a node (usually one already culled to a brick's region) that
/// calls no kernel.
pub fn lower(n: &SdfNode) -> Result<Program, LowerError> {
    lower_with(n, &[], 1.0)
}

/// Lowers a node whose kernels are in `kernels` (a volume's table: a
/// kernel's slot there is its index), each kernel's distance multiplied by
/// `kernel_scale`.
pub fn lower_with(n: &SdfNode, kernels: &[KernelSlot], kernel_scale: f32) -> Result<Program, LowerError> {
    let mut l = Lower { kernels, kernel_scale, out: vec![[0.0; 4]], payloads: Vec::new(), grids: Vec::new(), vdepth: 0, vmax: 0, pdepth: 0, pmax: 0, textures: 0 };
    l.node(n)?;
    if l.vmax > MAX_VALUE_DEPTH || l.pmax > MAX_POINT_DEPTH {
        return Err(LowerError::Unsupported("program exceeds GPU interpreter stack depth"));
    }
    let len = l.out.len();
    l.out[0] = [len as f32, l.vmax as f32, l.pmax as f32, 0.0];
    for (pc, data) in l.payloads {
        l.out[pc][1] = f32::from_bits(u32::try_from(l.out.len()).expect("program offset exceeds u32"));
        l.out[pc][2] = f32::from_bits(u32::try_from(data.len()).expect("links payload exceeds u32"));
        l.out.extend(data);
    }
    for (pc, data) in l.grids {
        l.out[pc + 1][0] = (l.out.len() - pc) as f32;
        l.out.extend(data);
    }
    Ok(Program { texels: l.out, value_depth: l.vmax, point_depth: l.pmax })
}

// LINKS payload (vec4s): radius/cell/clip-present/max cell count; clip plane;
// min; max; origin; dims; starts/cell-segs/segments offsets (scalar words).
// Arrays are packed four u32 bit patterns per texel. A segment is seven words:
// a.xyz, b.xyz, loop material slot. All offsets are relative to this payload.
fn links_data(l: &makepad_csg_sdf::Links) -> Vec<[f32; 4]> {
    let (cell, origin, dims, starts, cell_segs, segs) = l.grid();
    let bits = |n: usize| f32::from_bits(u32::try_from(n).expect("links index exceeds u32"));
    let (normal, off) = l.clip.unwrap_or(([0.0; 3], 0.0));
    let mut words = vec![
        f(l.radius), f(cell), if l.clip.is_some() { 1.0 } else { 0.0 },
        f32::from_bits(starts.windows(2).map(|s| s[1] - s[0]).max().unwrap_or(0)),
        f(normal[0]), f(normal[1]), f(normal[2]), f(off),
        f(l.min[0]), f(l.min[1]), f(l.min[2]), 0.0,
        f(l.max[0]), f(l.max[1]), f(l.max[2]), 0.0,
        f(origin[0]), f(origin[1]), f(origin[2]), 0.0,
        bits(dims[0]), bits(dims[1]), bits(dims[2]), 0.0,
        bits(28), bits(28 + starts.len()), bits(28 + starts.len() + cell_segs.len()), 0.0,
    ];
    words.extend(starts.iter().chain(cell_segs).map(|v| f32::from_bits(*v)));
    for (a, b, li) in segs {
        words.extend(a.iter().chain(b).map(|v| f(*v)));
        words.push(f32::from_bits(l.loops[*li as usize].slot as u32));
    }
    words.resize(words.len().div_ceil(4) * 4, 0.0);
    words.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}
