//! The fill's classes on the GPU (S1+S2, GPU-EXEC §9): the class program ([`regions_program`]) run
//! as compose rounds over a batch, from the outline's and the fill's fields.
//!
//! - A level-0 term reads its source directly: an input's FeatureSet with its field's sides, or an
//!   earlier round's output with its composed sides.
//! - A term at another level reads its source's level set: the extract (wall_pieces) cuts every
//!   level an input is read at in one run, and the term's sides come from the field's distances
//!   (d' <= level is inside).
//! - Flags read the earlier rounds' any-words.
//! - Every class comes out of a round: a class equal to an input runs one identity round.
//!
//! The extracts size their pieces at a checkpoint, so the rounds follow from their `then`: the
//! steps are continuation-passing, and [`steps`] hands the classes to its own `then`.
//!
//! Levels are mm in the program and grid units (0.1 mm = 1) on the GPU.
//!
//! [`regions_program`]: crate::regions_program
use crate::feature_field::FieldOut;
use crate::gpu_jobs::{CountWord, Counts, JobError, Pool, Step};
use crate::plane_abi::LEVEL_REGION;
use crate::regions_compose::{self as rc, ComposeKernels, RoundOut, RoundWords, Term, STATES};
use crate::regions_program::{Kind, Plane, Program, Source};
use crate::wall_pieces::{self, Levels, PieceKernels, PiecesOut};
use makepad_csg_sdf::plan::{Region, ShellSpec};
use makepad_csg_sdf::regions::{Reg, CLASSES};
use makepad_draw::Cx;

/// Grid units per mm.
const PER_MM: f32 = 10.0;
/// The pieces' level of a round that makes no class (a class's is LEVEL_REGION + its index in
/// CLASSES).
const LEVEL_INNER: u32 = LEVEL_REGION + CLASSES.len() as u32;
/// The extract's level id of level slot l (only the rounds read these pieces).
const LEVEL_SLOT: u32 = LEVEL_INNER + 1;

/// The batch: consecutive layers of one part of `n` layers, on an nx × ny coarse grid.
#[derive(Clone, Debug)]
pub struct Batch {
    pub nx: u32,
    pub ny: u32,
    pub n: usize,
    pub spec: ShellSpec,
    pub planes: Vec<Plane>,
}

/// An input region (the outline or the fill): its field, over the batch's planes, and the coarse
/// cells holding its segs. Its depths cover every level the program reads it at.
#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub field: FieldOut,
    pub cells: u32,
}

impl Input {
    /// The input's FeatureSet as a level-0 term at plane offset `dz`.
    fn term(&self, dz: i32) -> Term {
        let (i, f) = (&self.field.input, &self.field);
        Term::set(i.set_slot, i.seg_off, i.seg, f.band_slot, f.cover, f.node_side, dz)
    }
}

/// The class rounds' kernels, made once.
#[derive(Clone)]
pub struct ClassKernels {
    compose: ComposeKernels,
    pieces: PieceKernels,
}
impl ClassKernels {
    pub fn new(cx: &mut Cx) -> Result<Self, String> {
        Ok(Self { compose: ComposeKernels::new(cx)?, pieces: PieceKernels::new(cx)? })
    }
}

/// A class's pieces: output `output` of a round (its words, and its piece capacity).
#[derive(Clone, Copy, Debug)]
pub struct ClassRegion {
    pub out: RoundOut,
    pub output: u32,
    pub words: RoundWords,
    pub capacity: u32,
}

#[derive(Clone, Debug)]
pub struct ClassOut {
    /// Per class, in csg_sdf regions::CLASSES order: its pieces, or None when it is empty on
    /// every plane.
    pub classes: Vec<(Region, Option<ClassRegion>)>,
    /// Per batch plane: every class there reads only planes inside the batch.
    pub valid: Vec<bool>,
    /// Must all read 0 after the run; otherwise run again with a larger `factor`.
    pub overflow: Vec<CountWord>,
    /// The extracts the rounds read: check each with [`PiecesOut::verify`] on the final counts.
    pub extracts: Vec<PiecesOut>,
    /// Every program region a round made, in round order, for checks against a CPU run.
    pub regions: Vec<(Reg, ClassRegion)>,
}

/// A round already in the steps: its output, cell capacity and piece capacity.
#[derive(Clone, Copy)]
struct Made {
    out: RoundOut,
    words: RoundWords,
    cells: u32,
    outputs: u32,
}

/// The levels one input is read at: per slot, the level on every field plane (grid units, NaN
/// where unused), and that input's extract once it ran.
#[derive(Default)]
struct Read {
    slots: Vec<Vec<f32>>,
    pieces: Option<PiecesOut>,
}

impl Read {
    /// The slot of a level vector, added when new.
    fn slot(&mut self, levels: Vec<f32>) -> u32 {
        let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        match self.slots.iter().position(|s| same(s, &levels)) {
            Some(i) => i as u32,
            None => {
                self.slots.push(levels);
                self.slots.len() as u32 - 1
            }
        }
    }
}

/// What the rounds need, owned so the steps can carry it past each checkpoint.
struct Run {
    k: ClassKernels,
    batch: Batch,
    prog: Program,
    inputs: [Input; 2],
    reads: [Read; 2],
    factor: u32,
    then: Box<dyn FnOnce(&mut Pool, &ClassOut) -> Result<Vec<Step>, JobError>>,
}

fn which(s: Source) -> Option<usize> {
    match s {
        Source::Outline => Some(0),
        Source::Fill => Some(1),
        _ => None,
    }
}

/// A level term's levels on its field's planes: field plane q holds the level of reading plane
/// q - dz (grid units), NaN off the batch.
fn field_levels(level: &[f32], dz: i8) -> Vec<f32> {
    let planes = level.len() as i64;
    (0..planes).map(|q| {
        let p = q - dz as i64;
        if (0..planes).contains(&p) { level[p as usize] * PER_MM } else { f32::NAN }
    }).collect()
}

/// The class rounds of `batch`, from `outline` and `fill`; `then` gets the classes and its steps
/// run after theirs. Every round holds `factor` times its terms' segs in pieces.
pub fn steps(k: &ClassKernels, pool: &mut Pool, batch: &Batch, outline: Input, fill: Input, factor: u32,
    then: impl FnOnce(&mut Pool, &ClassOut) -> Result<Vec<Step>, JobError> + 'static) -> Result<Vec<Step>, JobError> {
    let fail = JobError::Failed;
    let prog = Program::compile(&batch.spec, batch.n, &batch.planes).map_err(fail)?;
    let planes = batch.planes.len() as u32;
    let inputs = [outline, fill];
    for (i, inp) in inputs.iter().enumerate() {
        let p = &inp.field.params;
        if p.planes != planes || p.nx != batch.nx || p.ny != batch.ny {
            return Err(fail(format!("input {i}: field {}x{}x{} for a {}x{}x{planes} batch", p.nx, p.ny, p.planes, batch.nx, batch.ny)));
        }
    }
    // every level an input is read at, so each input runs one extract
    let mut reads = [Read::default(), Read::default()];
    for (i, r) in prog.rounds.iter().enumerate() {
        let Kind::Compose { terms, .. } = &r.kind else { continue };
        for t in terms.iter().filter(|t| t.level.iter().any(|&l| l != 0.0)) {
            let Some(w) = which(t.src) else {
                return Err(fail(format!("round {i} reads {:?} at a level: that needs the region field stage", t.src)));
            };
            let levels = field_levels(&t.level, t.dz);
            let p = &inputs[w].field.params;
            if let Some(l) = levels.iter().find(|l| **l < -p.depth_in || **l > p.depth_out) {
                return Err(fail(format!("{:?} is read at level {l} past its field's depths ({}, {})", t.src, p.depth_in, p.depth_out)));
            }
            reads[w].slot(levels);
        }
    }
    let run = Run { k: k.clone(), batch: batch.clone(), prog, inputs, reads, factor, then: Box::new(then) };
    extract(run, pool)
}

/// The next input's extract, then the rounds from its `then`.
fn extract(mut run: Run, pool: &mut Pool) -> Result<Vec<Step>, JobError> {
    let Some(w) = (0..2).find(|&w| !run.reads[w].slots.is_empty() && run.reads[w].pieces.is_none()) else {
        return rounds(run, pool);
    };
    let read = &run.reads[w];
    let slots = read.slots.len();
    let planes = run.batch.planes.len();
    let ids: Vec<u32> = (0..slots as u32).map(|l| LEVEL_SLOT + l).collect();
    let values: Vec<f32> = (0..planes).flat_map(|q| read.slots.iter().map(move |s| s[q])).collect();
    let words = wall_pieces::claim(pool)?;
    let (k, field) = (run.k.pieces.clone(), run.inputs[w].field);
    wall_pieces::steps(&k, pool, &field, words, &Levels { ids: &ids, values: &values }, move |_: &Counts, pool: &mut Pool, out: &PiecesOut| {
        run.reads[w].pieces = Some(*out);
        extract(run, pool)
    })
}

/// Every round, the classes, then the caller's steps.
fn rounds(run: Run, pool: &mut Pool) -> Result<Vec<Step>, JobError> {
    let fail = JobError::Failed;
    let Run { k, batch, prog, inputs, reads, factor, then } = run;
    let planes = batch.planes.len() as u32;
    let plane_cells = planes * batch.nx * batch.ny;
    let level_of = |out: Option<Reg>| {
        out.and_then(|r| CLASSES.iter().position(|c| c.1 == r)).map_or(LEVEL_INNER, |c| LEVEL_REGION + c as u32)
    };
    let mut steps = Vec::new();
    let mut made: Vec<Made> = Vec::new();
    let mut regions = Vec::new();
    let mut overflow = Vec::new();
    // One compose round over `terms` (with their cell and seg capacities), its tables and flags.
    let mut round = |pool: &mut Pool, steps: &mut Vec<Step>, terms: Vec<(Term, u32, u32)>, tables: Vec<[[u32; 2]; STATES]>, flags: Vec<rc::Flag>, level: u32| -> Result<Made, JobError> {
        let cells = terms.iter().map(|t| t.1).sum::<u32>().min(plane_cells);
        let segs = terms.iter().map(|t| t.2).sum::<u32>();
        let words = RoundWords::claim(pool, "regions class round", planes, 1)?;
        let r = rc::Round { nx: batch.nx, ny: batch.ny, planes, level, terms: terms.into_iter().map(|t| t.0).collect(), flags, tables: vec![tables],
            cells, segs, outputs: segs.max(1) * factor };
        let (s, out) = rc::steps(&k.compose, pool, &r, words).map_err(fail)?;
        steps.extend(s);
        overflow.push(words.overflow);
        Ok(Made { out, words, cells: r.cells, outputs: r.outputs })
    };
    let input_term = |s: Source, dz: i8| which(s).map(|w| (inputs[w].term(dz as i32), inputs[w].cells, inputs[w].field.input.segs.capacity()));
    for (i, r) in prog.rounds.iter().enumerate() {
        let Kind::Compose { terms, tables } = &r.kind else {
            return Err(fail(format!("round {i} ({:?}) keeps components: that needs the chain stage", r.out)));
        };
        let mut cterms = Vec::new();
        for t in terms {
            if t.level.iter().all(|&l| l == 0.0) {
                cterms.push(match t.src {
                    Source::Round(j) => (made[j].out.term(0, t.dz as i32), made[j].cells, made[j].outputs),
                    s => input_term(s, t.dz).ok_or_else(|| fail(format!("round {i} reads {s:?}")))?,
                });
            } else {
                // the extract ran in `steps`: every input level term has its slot
                let w = which(t.src).expect("an input");
                let read = &reads[w];
                let slots = read.slots.len() as u32;
                let levels = field_levels(&t.level, t.dz);
                let slot = read.slots.iter().position(|s| s.iter().zip(&levels).all(|(a, b)| a.to_bits() == b.to_bits())).expect("a slot") as u32;
                let (f, p) = (&inputs[w].field, read.pieces.expect("the extract ran"));
                let reading: Vec<f32> = t.level.iter().map(|l| l * PER_MM).collect();
                cterms.push((Term::level(f.band_slot, f.cover, f.nf, p.piece_off, p.piece, slots, slot, t.dz as i32, reading), f.slots, p.capacity));
            }
        }
        let flags = r.flags.iter().map(|f| made[f.of].out.flag(0, f.dz as i32)).collect();
        let m = round(pool, &mut steps, cterms, tables.clone(), flags, level_of(r.out))?;
        if let Some(reg) = r.out {
            regions.push((reg, ClassRegion { out: m.out, output: 0, words: m.words, capacity: m.outputs }));
        }
        made.push(m);
    }
    let mut classes = Vec::new();
    let mut valid = vec![true; planes as usize];
    for (c, (region, reg)) in CLASSES.iter().enumerate() {
        let src = prog.regions.get(reg).copied().ok_or_else(|| fail(format!("{reg:?} is not made")))?;
        let class = match src {
            Source::Empty => None,
            Source::Round(j) => {
                for (p, v) in valid.iter_mut().enumerate() { *v &= prog.valid[j][p]; }
                Some(ClassRegion { out: made[j].out, output: 0, words: made[j].words, capacity: made[j].outputs })
            }
            s => {
                // an input as is: one identity round brings it into the class pieces' layout
                let term = input_term(s, 0).ok_or_else(|| fail(format!("{reg:?} is {s:?}")))?;
                let identity = rc::table(1, |b| b == 1);
                let m = round(pool, &mut steps, vec![term], vec![[identity; STATES]; planes as usize], Vec::new(), LEVEL_REGION + c as u32)?;
                Some(ClassRegion { out: m.out, output: 0, words: m.words, capacity: m.outputs })
            }
        };
        classes.push((*region, class));
    }
    let extracts = reads.iter().filter_map(|r| r.pieces).collect();
    steps.extend(then(pool, &ClassOut { classes, valid, overflow, extracts, regions })?);
    Ok(steps)
}
