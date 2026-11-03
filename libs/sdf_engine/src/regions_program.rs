//! The class program (csg_sdf::regions) as rounds over a batch of planes (S1+S2).
//!
//! A batch holds consecutive layers of one part, each with its level set. Every program round
//! becomes one batch round with one term list for all planes: (region, plane offset) pairs, each
//! read at its own level per plane. A plane's operation, its variants at the part's ends and its
//! IfAny conditions become per-plane truth tables over those terms, one per flag state; at most
//! [`MAX_FLAGS`] flags pick the state on the GPU (a flag: an earlier round's region has a piece on
//! plane p + dz). A union of more than [`MAX_TERMS`] terms becomes a chain of rounds. A Keep round
//! filters one region's components, with a mode per plane and flag state.
//!
//! A region that equals an input or an earlier round on every plane costs no round (see
//! [`Program::regions`]), and a term on an empty region folds into the tables.
//!
//! Halo planes run the same rounds. [`Program::valid`] names, per round, the planes whose result
//! reads only planes inside the batch, through every earlier round.
use crate::regions_compose::{table, MAX_FLAGS, MAX_TERMS, STATES};
use makepad_csg_sdf::plan::{ShellLevels, ShellSpec};
use makepad_csg_sdf::regions::{self, Expr, Op, Reg, Src, Test};
use std::collections::HashMap;

/// A region a round reads: an input, an earlier round's output, or nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    Outline,
    Fill,
    Round(usize),
    Empty,
}

/// A term of a compose round: `src` on plane p + dz at `level[p]` mm (positive grows; 0 on a
/// plane that does not read it).
#[derive(Clone, Debug, PartialEq)]
pub struct Term {
    pub src: Source,
    pub dz: i8,
    pub level: Vec<f32>,
}

/// A flag: round `of` has a piece on plane p + dz (off the batch: none).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Flag {
    pub of: usize,
    pub dz: i8,
}

/// What a Keep round keeps of a plane's components: none, all, or those for which every test
/// holds (`Hold(true)`) or not every test holds (`Hold(false)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Nothing,
    All,
    Hold(bool),
}

/// A per-component test of a Keep round, its constant per plane (0 where a plane keeps no
/// component by tests).
#[derive(Clone, Debug, PartialEq)]
pub enum KeepTest {
    AreaOver(Vec<f32>),
    Thin(Vec<f32>),
    Meets(Source, i8),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    /// Per plane and flag state, a truth table over `terms` ([`table`]).
    Compose { terms: Vec<Term>, tables: Vec<[[u32; 2]; STATES]> },
    /// Per plane and flag state, a mode over the components of `of`.
    Keep { of: Source, tests: Vec<KeepTest>, modes: Vec<[Mode; STATES]> },
}

/// One round of the batch: the program region it makes (none for a link of a union chain).
#[derive(Clone, Debug, PartialEq)]
pub struct Round {
    pub out: Option<Reg>,
    pub flags: Vec<Flag>,
    pub kind: Kind,
}

/// A batch plane: its layer in the part and its level set.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub k: usize,
    pub levels: ShellLevels,
}

#[derive(Clone, Debug, Default)]
pub struct Program {
    pub rounds: Vec<Round>,
    /// Where every program region is: a round, or the input or earlier round it equals.
    pub regions: HashMap<Reg, Source>,
    /// Per round and batch plane: its result reads only planes inside the batch.
    pub valid: Vec<Vec<bool>>,
}

/// A plane's composition over the round's terms.
#[derive(Clone, Debug)]
enum X {
    Const(bool),
    Term(usize, bool),
    All(Vec<X>),
    Any(Vec<X>),
}

impl X {
    fn eval(&self, bits: u32) -> bool {
        match self {
            X::Const(v) => *v,
            X::Term(t, neg) => (bits >> t & 1 != 0) != *neg,
            X::All(of) => of.iter().all(|x| x.eval(bits)),
            X::Any(of) => of.iter().any(|x| x.eval(bits)),
        }
    }
    /// Adds the terms of a union of positive terms to `out`; false when it is not one.
    fn union(&self, out: &mut Vec<usize>) -> bool {
        match self {
            X::Const(false) => true,
            X::Term(t, false) => {
                out.push(*t);
                true
            }
            X::Any(of) => of.iter().all(|x| x.union(out)),
            _ => false,
        }
    }
}

/// What a plane reads in one round: (source, plane offset) pairs and flags.
#[derive(Default)]
struct Reads {
    sources: Vec<(Source, i8)>,
    flags: Vec<usize>,
}

impl Program {
    /// The program of `spec` for the batch `planes` (consecutive layers of a part of `n`).
    pub fn compile(spec: &ShellSpec, n: usize, planes: &[Plane]) -> Result<Self, String> {
        let mut prog = Program::default();
        let Some(first) = planes.first() else { return Ok(prog) };
        if planes.iter().enumerate().any(|(i, p)| p.k != first.k + i || p.k >= n) {
            return Err("batch planes are not consecutive layers of the part".into());
        }
        let plans: Vec<Vec<regions::Round>> = planes.iter().map(|p| regions::rounds(spec, &p.levels, p.k, n)).collect();
        for i in 0..plans[0].len() {
            let out = plans[0][i].out;
            if plans.iter().any(|plan| plan[i].out != out) {
                return Err(format!("round {i}: planes make different regions"));
            }
            let ops: Vec<&Op> = plans.iter().map(|plan| &plan[i].op).collect();
            prog.round(out, &ops)?;
        }
        Ok(prog)
    }

    fn source(&self, src: Src) -> Result<Source, String> {
        match src {
            Src::Outline => Ok(Source::Outline),
            Src::Fill => Ok(Source::Fill),
            Src::Reg(r) => self.regions.get(&r).copied().ok_or_else(|| format!("{r:?} read before it is made")),
        }
    }

    fn valid_at(&self, s: Source, p: i64, planes: usize) -> bool {
        (0..planes as i64).contains(&p) && match s {
            Source::Round(j) => self.valid[j][p as usize],
            _ => true,
        }
    }

    /// The flags of every plane's IfAny conditions, an empty region's condition folded away.
    fn flags(&self, ops: &[&Op]) -> Result<Vec<Flag>, String> {
        fn walk(prog: &Program, op: &Op, out: &mut Vec<Flag>) -> Result<(), String> {
            if let Op::IfAny { of, dz, then, otherwise } = op {
                match prog.source(Src::Reg(*of))? {
                    Source::Round(j) => {
                        let f = Flag { of: j, dz: *dz };
                        if !out.contains(&f) {
                            out.push(f);
                        }
                    }
                    Source::Empty => {}
                    s => return Err(format!("an IfAny on the input {s:?}")),
                }
                walk(prog, then, out)?;
                walk(prog, otherwise, out)?;
            }
            Ok(())
        }
        let mut out = Vec::new();
        for op in ops {
            walk(self, op, &mut out)?;
        }
        if out.len() > MAX_FLAGS {
            return Err(format!("{} IfAny conditions in one round", out.len()));
        }
        Ok(out)
    }

    /// The operation a plane runs in flag state `state`, and the flags it read on the way.
    fn leaf<'a>(&self, op: &'a Op, flags: &[Flag], state: usize, read: &mut Vec<usize>) -> Result<&'a Op, String> {
        match op {
            Op::IfAny { of, dz, then, otherwise } => {
                let on = match self.source(Src::Reg(*of))? {
                    Source::Round(j) => {
                        let f = flags.iter().position(|f| *f == Flag { of: j, dz: *dz }).expect("collected");
                        if !read.contains(&f) {
                            read.push(f);
                        }
                        state >> f & 1 != 0
                    }
                    _ => false,
                };
                self.leaf(if on { then } else { otherwise }, flags, state, read)
            }
            op => Ok(op),
        }
    }

    fn round(&mut self, out: Reg, ops: &[&Op]) -> Result<(), String> {
        let planes = ops.len();
        let flags = self.flags(ops)?;
        let mut leaves: Vec<[&Op; STATES]> = Vec::with_capacity(planes);
        let mut reads: Vec<Reads> = (0..planes).map(|_| Reads::default()).collect();
        for (p, op) in ops.iter().enumerate() {
            let mut l = [&Op::Empty; STATES];
            for (s, l) in l.iter_mut().enumerate() {
                *l = self.leaf(op, &flags, s, &mut reads[p].flags)?;
            }
            leaves.push(l);
        }
        let keep = leaves.iter().flatten().any(|op| matches!(op, Op::Keep { .. }));
        if keep { self.keep(out, flags, &leaves, reads) } else { self.compose(out, flags, &leaves, reads) }
    }

    /// Validity per plane from what each plane reads.
    fn validity(&self, flags: &[Flag], reads: &[Reads]) -> Vec<bool> {
        let planes = reads.len();
        reads.iter().enumerate().map(|(p, r)| {
            r.sources.iter().all(|&(s, dz)| self.valid_at(s, p as i64 + dz as i64, planes))
                && r.flags.iter().all(|&f| self.valid_at(Source::Round(flags[f].of), p as i64 + flags[f].dz as i64, planes))
        }).collect()
    }

    fn push(&mut self, out: Option<Reg>, flags: Vec<Flag>, kind: Kind, valid: Vec<bool>) -> Source {
        self.rounds.push(Round { out, flags, kind });
        self.valid.push(valid);
        Source::Round(self.rounds.len() - 1)
    }

    fn compose(&mut self, out: Reg, flags: Vec<Flag>, leaves: &[[&Op; STATES]], mut reads: Vec<Reads>) -> Result<(), String> {
        let planes = leaves.len();
        let mut keys: Vec<(Source, i8)> = Vec::new();
        let mut levels: Vec<Vec<Option<f32>>> = Vec::new();
        let mut xs: Vec<[X; STATES]> = Vec::with_capacity(planes);
        for (p, l) in leaves.iter().enumerate() {
            let mut term = |src: Src, dz: i8, level: f32, neg: bool, reads: &mut Reads| -> Result<X, String> {
                let s = self.source(src)?;
                if s == Source::Empty {
                    return Ok(X::Const(neg));
                }
                let t = match keys.iter().position(|k| *k == (s, dz)) {
                    Some(t) => t,
                    None => {
                        keys.push((s, dz));
                        levels.push(vec![None; planes]);
                        keys.len() - 1
                    }
                };
                match levels[t][p] {
                    Some(at) if at != level => return Err(format!("{out:?}: plane {p} reads {s:?} at {dz} at two levels")),
                    _ => levels[t][p] = Some(level),
                }
                if !reads.sources.contains(&(s, dz)) {
                    reads.sources.push((s, dz));
                }
                Ok(X::Term(t, neg))
            };
            fn expr(e: &Expr, term: &mut dyn FnMut(Src, i8, f32, bool) -> Result<X, String>) -> Result<X, String> {
                Ok(match e {
                    Expr::Term(t) => term(t.src, t.dz, t.level, t.neg)?,
                    Expr::All(of) => X::All(of.iter().map(|e| expr(e, term)).collect::<Result<_, _>>()?),
                    Expr::Any(of) => X::Any(of.iter().map(|e| expr(e, term)).collect::<Result<_, _>>()?),
                })
            }
            let r = &mut reads[p];
            let mut term = |src: Src, dz: i8, level: f32, neg: bool| term(src, dz, level, neg, r);
            let mut x: [X; STATES] = std::array::from_fn(|_| X::Const(false));
            for (s, op) in l.iter().enumerate() {
                x[s] = match op {
                    Op::Empty => X::Const(false),
                    Op::Same(src) => term(*src, 0, 0.0, false)?,
                    Op::Compose(e) => expr(e, &mut term)?,
                    Op::Concat(of) => X::Any(of.iter().map(|r| term(Src::Reg(*r), 0, 0.0, false)).collect::<Result<_, _>>()?),
                    op => return Err(format!("{out:?}: {op:?} in a composition")),
                };
            }
            xs.push(x);
        }
        let terms: Vec<Term> = keys.iter().zip(&levels).map(|(&(src, dz), l)| Term { src, dz, level: l.iter().map(|l| l.unwrap_or(0.0)).collect() }).collect();
        let made = if terms.len() <= MAX_TERMS as usize {
            let n = terms.len() as u32;
            let tables: Vec<[[u32; 2]; STATES]> = xs.iter().map(|x| std::array::from_fn(|s| table(n, |b| x[s].eval(b)))).collect();
            if tables.iter().flatten().all(|t| *t == [0, 0]) {
                Source::Empty
            } else if n == 1 && terms[0].dz == 0 && terms[0].level.iter().all(|&l| l == 0.0) && tables.iter().flatten().all(|t| *t == [0b10, 0]) {
                terms[0].src
            } else {
                let valid = self.validity(&flags, &reads);
                self.push(Some(out), flags, Kind::Compose { terms, tables }, valid)
            }
        } else {
            // A long union: links of at most MAX_TERMS terms, each carrying the one before.
            let sets: Vec<[Vec<usize>; STATES]> = xs.iter().map(|x| {
                let mut sets: [Vec<usize>; STATES] = Default::default();
                for (s, set) in sets.iter_mut().enumerate() {
                    if !x[s].union(set) {
                        return Err(format!("{out:?}: over {MAX_TERMS} terms and not a union"));
                    }
                }
                Ok(sets)
            }).collect::<Result<_, String>>()?;
            let mut prev: Option<Source> = None;
            let mut at = 0;
            loop {
                let carry = prev.is_some() as usize;
                let end = (at + MAX_TERMS as usize - carry).min(terms.len());
                let mut link: Vec<Term> = prev.map(|s| Term { src: s, dz: 0, level: vec![0.0; planes] }).into_iter().collect();
                link.extend(terms[at..end].iter().cloned());
                let n = link.len() as u32;
                let tables = sets.iter().map(|set| std::array::from_fn(|s| table(n, |b| {
                    (carry == 1 && b & 1 != 0) || set[s].iter().any(|&t| (at..end).contains(&t) && b >> (t - at + carry) & 1 != 0)
                }))).collect();
                let link_reads: Vec<Reads> = (0..planes).map(|p| Reads {
                    sources: terms[at..end].iter().map(|t| (t.src, t.dz)).filter(|k| reads[p].sources.contains(k)).chain(prev.map(|s| (s, 0))).collect(),
                    flags: reads[p].flags.clone(),
                }).collect();
                let valid = self.validity(&flags, &link_reads);
                let last = end == terms.len();
                prev = Some(self.push(last.then_some(out), flags.clone(), Kind::Compose { terms: link, tables }, valid));
                if last {
                    break prev.unwrap();
                }
                at = end;
            }
        };
        self.regions.insert(out, made);
        Ok(())
    }

    fn keep(&mut self, out: Reg, flags: Vec<Flag>, leaves: &[[&Op; STATES]], mut reads: Vec<Reads>) -> Result<(), String> {
        let planes = leaves.len();
        let mut of: Option<Source> = None;
        let mut tests: Vec<KeepTest> = Vec::new();
        let mut never = false;
        for (p, l) in leaves.iter().enumerate() {
            let mut shaped = false;
            for op in l.iter() {
                let Op::Keep { of: o, tests: t, .. } = op else { continue };
                let o = self.source(Src::Reg(*o))?;
                if of.is_some_and(|of| of != o) {
                    return Err(format!("{out:?}: keeps two regions"));
                }
                of = Some(o);
                if tests.is_empty() {
                    tests = t.iter().map(|t| match *t {
                        Test::AreaOver(_) => Ok(KeepTest::AreaOver(vec![0.0; planes])),
                        Test::Thin(_) => Ok(KeepTest::Thin(vec![0.0; planes])),
                        Test::Meets(src, dz) => Ok(KeepTest::Meets(self.source(src)?, dz)),
                    }).collect::<Result<_, String>>()?;
                }
                if t.len() != tests.len() {
                    return Err(format!("{out:?}: test lists differ between planes"));
                }
                if shaped {
                    continue;
                }
                shaped = true;
                for (into, t) in tests.iter_mut().zip(t) {
                    match (into, *t) {
                        (KeepTest::AreaOver(v), Test::AreaOver(a)) => v[p] = a,
                        (KeepTest::Thin(v), Test::Thin(r)) => v[p] = r,
                        (KeepTest::Meets(s, d), Test::Meets(src, dz)) if *s == self.source(src)? && *d == dz => {
                            reads[p].sources.push((*s, dz));
                            never |= *s == Source::Empty;
                        }
                        _ => return Err(format!("{out:?}: test lists differ between planes")),
                    }
                }
            }
        }
        let of = of.expect("a Keep leaf");
        let modes: Vec<[Mode; STATES]> = leaves.iter().map(|l| {
            let mut m = [Mode::Nothing; STATES];
            for (s, op) in l.iter().enumerate() {
                m[s] = match op {
                    Op::Empty => Mode::Nothing,
                    Op::Same(src) if self.source(*src)? == of => Mode::All,
                    // a test that never holds: every component fails it
                    Op::Keep { hold, .. } if never => if *hold { Mode::Nothing } else { Mode::All },
                    Op::Keep { hold, .. } => Mode::Hold(*hold),
                    op => return Err(format!("{out:?}: {op:?} beside a Keep")),
                };
            }
            Ok(m)
        }).collect::<Result<_, String>>()?;
        for (p, m) in modes.iter().enumerate() {
            if m.iter().any(|m| *m != Mode::Nothing) {
                reads[p].sources.push((of, 0));
            }
        }
        let made = if of == Source::Empty || modes.iter().flatten().all(|m| *m == Mode::Nothing) {
            Source::Empty
        } else if modes.iter().flatten().all(|m| *m == Mode::All) {
            of
        } else {
            if never {
                tests.clear();
            }
            let valid = self.validity(&flags, &reads);
            self.push(Some(out), flags, Kind::Keep { of, tests, modes }, valid)
        };
        self.regions.insert(out, made);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every operation on one line of samples per plane: booleans per sample, an offset as a
    /// window, components as runs. The program run directly and as batch rounds must agree on
    /// every plane the batch calls valid.
    const S: usize = 48;
    const UNIT: f32 = 0.1;
    type Line = Vec<bool>;

    fn offset(x: &[bool], level: f32) -> Line {
        let k = (level / UNIT).round() as i64;
        let at = |j: i64| (0..S as i64).contains(&j) && x[j as usize];
        (0..S as i64).map(|i| if k >= 0 { (i - k..=i + k).any(at) } else { (i + k..=i - k).all(at) }).collect()
    }

    fn runs(x: &[bool]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < S {
            if x[i] {
                let a = i;
                while i < S && x[i] {
                    i += 1;
                }
                out.push((a, i));
            } else {
                i += 1;
            }
        }
        out
    }

    fn only(r: (usize, usize)) -> Line {
        (0..S).map(|i| (r.0..r.1).contains(&i)).collect()
    }

    fn levels(scale: f32) -> ShellLevels {
        let s = |v: f32| v * scale;
        ShellLevels { speck: s(0.15), face_grow: s(0.2), bottom_grow: s(0.3), shell_open: s(0.2), min_sparse_area: s(0.6), vertical: s(0.25), bridge_open: s(0.15), bridge_area: s(0.3), bridge_grow: s(0.3), under_grow: s(0.1), band_core: s(0.2), band_area: s(0.2), adjacent: s(0.15), narrow: s(0.4) }
    }

    fn spec(t: u32, b: u32, flags: bool) -> ShellSpec {
        ShellSpec { top_layers: t, bottom_layers: b, vertical_shells: flags, floating_shells: flags, narrow_solid: flags, sets: vec![levels(1.0), levels(1.5)] }
    }

    /// Layer k's level set: the second set on every third layer.
    fn set(spec: &ShellSpec, k: usize) -> ShellLevels {
        spec.sets[usize::from(k % 3 == 2)]
    }

    struct Part {
        outline: Vec<Line>,
        fill: Vec<Line>,
    }

    fn part(n: usize, seed: u64) -> Part {
        let mut s = seed;
        let mut rnd = |m: usize| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as usize % m
        };
        let outline: Vec<Line> = (0..n).map(|_| {
            let mut l = vec![false; S];
            for _ in 0..1 + rnd(3) {
                let a = rnd(S - 4);
                let w = 3 + rnd(S / 2);
                for v in l.iter_mut().skip(a).take(w) {
                    *v = true;
                }
            }
            l
        }).collect();
        let fill = outline.iter().map(|o| offset(o, -0.2)).collect();
        Part { outline, fill }
    }

    fn holds(t: &Test, c: (usize, usize), made: &[HashMap<Reg, Line>], part: &Part, k: usize) -> bool {
        match *t {
            Test::AreaOver(a) => (c.1 - c.0) as f32 * UNIT > a,
            Test::Thin(r) => !offset(&only(c), -r).contains(&true),
            Test::Meets(src, dz) => {
                let other = region(src, made, part, (k as i64 + dz as i64) as usize);
                (c.0..c.1).any(|i| other[i])
            }
        }
    }

    fn region<'a>(src: Src, made: &'a [HashMap<Reg, Line>], part: &'a Part, k: usize) -> &'a Line {
        match src {
            Src::Outline => &part.outline[k],
            Src::Fill => &part.fill[k],
            Src::Reg(r) => &made[k][&r],
        }
    }

    /// The program on every plane of the part, round by round (as the planner's chain runs).
    fn direct(spec: &ShellSpec, part: &Part) -> Vec<HashMap<Reg, Line>> {
        let n = part.outline.len();
        let plans: Vec<Vec<regions::Round>> = (0..n).map(|k| regions::rounds(spec, &set(spec, k), k, n)).collect();
        let mut made: Vec<HashMap<Reg, Line>> = vec![HashMap::new(); n];
        fn expr(e: &Expr, made: &[HashMap<Reg, Line>], part: &Part, k: usize) -> Line {
            match e {
                Expr::Term(t) => {
                    let r = offset(region(t.src, made, part, (k as i64 + t.dz as i64) as usize), t.level);
                    if t.neg { r.iter().map(|v| !v).collect() } else { r }
                }
                Expr::All(of) => of.iter().fold(vec![true; S], |acc, e| acc.iter().zip(expr(e, made, part, k)).map(|(a, b)| *a && b).collect()),
                Expr::Any(of) => of.iter().fold(vec![false; S], |acc, e| acc.iter().zip(expr(e, made, part, k)).map(|(a, b)| *a || b).collect()),
            }
        }
        fn op(o: &Op, made: &[HashMap<Reg, Line>], part: &Part, k: usize) -> Line {
            match o {
                Op::Empty => vec![false; S],
                Op::Same(src) => region(*src, made, part, k).clone(),
                Op::Compose(e) => expr(e, made, part, k),
                Op::Keep { of, tests, hold } => {
                    let mut out = vec![false; S];
                    for c in runs(&made[k][of]) {
                        if tests.iter().all(|t| holds(t, c, made, part, k)) == *hold {
                            out[c.0..c.1].iter_mut().for_each(|v| *v = true);
                        }
                    }
                    out
                }
                Op::Concat(of) => (0..S).map(|i| of.iter().any(|r| made[k][r][i])).collect(),
                Op::IfAny { of, dz, then, otherwise } => {
                    let any = made[(k as i64 + *dz as i64) as usize][of].contains(&true);
                    op(if any { then } else { otherwise }, made, part, k)
                }
            }
        }
        for i in 0..plans[0].len() {
            let lines: Vec<Line> = (0..n).map(|k| op(&plans[k][i].op, &made, part, k)).collect();
            for (k, l) in lines.into_iter().enumerate() {
                made[k].insert(plans[k][i].out, l);
            }
        }
        made
    }

    /// The batch rounds on planes k0.. of the part.
    fn batched(prog: &Program, part: &Part, k0: usize, planes: usize) -> Vec<Vec<Line>> {
        let mut vals: Vec<Vec<Line>> = Vec::new();
        let value = |vals: &Vec<Vec<Line>>, s: Source, q: i64| -> Line {
            if !(0..planes as i64).contains(&q) {
                return vec![false; S];
            }
            let q = q as usize;
            match s {
                Source::Outline => part.outline[k0 + q].clone(),
                Source::Fill => part.fill[k0 + q].clone(),
                Source::Round(j) => vals[j][q].clone(),
                Source::Empty => vec![false; S],
            }
        };
        for r in &prog.rounds {
            let lines: Vec<Line> = (0..planes).map(|p| {
                let state: usize = r.flags.iter().enumerate()
                    .map(|(i, f)| usize::from(value(&vals, Source::Round(f.of), p as i64 + f.dz as i64).contains(&true)) << i).sum();
                match &r.kind {
                    Kind::Compose { terms, tables } => {
                        let at: Vec<Line> = terms.iter().map(|t| offset(&value(&vals, t.src, p as i64 + t.dz as i64), t.level[p])).collect();
                        let [lo, hi] = tables[p][state];
                        (0..S).map(|i| {
                            let b: u32 = at.iter().enumerate().map(|(t, l)| u32::from(l[i]) << t).sum();
                            (if b < 32 { lo >> b } else { hi >> (b - 32) }) & 1 != 0
                        }).collect()
                    }
                    Kind::Keep { of, tests, modes } => {
                        let of = value(&vals, *of, p as i64);
                        let mut out = vec![false; S];
                        for c in runs(&of) {
                            let pass = tests.iter().all(|t| match t {
                                KeepTest::AreaOver(a) => (c.1 - c.0) as f32 * UNIT > a[p],
                                KeepTest::Thin(r) => !offset(&only(c), -r[p]).contains(&true),
                                KeepTest::Meets(s, dz) => {
                                    let other = value(&vals, *s, p as i64 + *dz as i64);
                                    (c.0..c.1).any(|i| other[i])
                                }
                            });
                            let kept = match modes[p][state] {
                                Mode::Nothing => false,
                                Mode::All => true,
                                Mode::Hold(h) => pass == h,
                            };
                            if kept {
                                out[c.0..c.1].iter_mut().for_each(|v| *v = true);
                            }
                        }
                        out
                    }
                }
            }).collect();
            vals.push(lines);
        }
        vals
    }

    #[test]
    fn batch_rounds_run_the_program() {
        let mut rounds_max = 0;
        for (t, b, flags) in [(1, 1, false), (4, 3, true), (8, 7, true)] {
            let spec = spec(t, b, flags);
            let n = 14;
            for seed in 0..2 {
                let part = part(n, seed * 31 + t as u64);
                let want = direct(&spec, &part);
                for (k0, planes) in [(0, n), (3, 9), (2, 1)] {
                    let batch: Vec<Plane> = (k0..k0 + planes).map(|k| Plane { k, levels: set(&spec, k) }).collect();
                    let prog = Program::compile(&spec, n, &batch).unwrap();
                    rounds_max = rounds_max.max(prog.rounds.len());
                    for r in &prog.rounds {
                        assert!(r.flags.len() <= MAX_FLAGS);
                        if let Kind::Compose { terms, .. } = &r.kind {
                            assert!(terms.len() <= MAX_TERMS as usize);
                        }
                    }
                    let got = batched(&prog, &part, k0, planes);
                    for (reg, src) in &prog.regions {
                        for p in 0..planes {
                            let valid = match src {
                                Source::Round(j) => prog.valid[*j][p],
                                _ => true,
                            };
                            if !valid {
                                continue;
                            }
                            let ours = match src {
                                Source::Outline => part.outline[k0 + p].clone(),
                                Source::Fill => part.fill[k0 + p].clone(),
                                Source::Round(j) => got[*j][p].clone(),
                                Source::Empty => vec![false; S],
                            };
                            assert_eq!(ours, want[k0 + p][reg], "spec ({t}, {b}, {flags}) seed {seed} batch {k0}+{planes}: {reg:?} on layer {}", k0 + p);
                        }
                    }
                    // the whole part is valid everywhere
                    if planes == n {
                        assert!(prog.valid.iter().flatten().all(|v| *v));
                    }
                }
            }
        }
        assert!(rounds_max > 0);
    }

    /// The window decides the halo: a class on a batch plane is valid once the batch holds the
    /// planes its rounds read, and the middle of a batch with that halo is valid.
    #[test]
    fn classes_are_valid_inside_the_halo() {
        let spec = spec(4, 3, true);
        let n = 30;
        let planes = 21;
        let batch: Vec<Plane> = (5..5 + planes).map(|k| Plane { k, levels: set(&spec, k) }).collect();
        let prog = Program::compile(&spec, n, &batch).unwrap();
        let valid: Vec<usize> = (0..planes).filter(|&p| regions::CLASSES.iter().all(|(_, r)| match prog.regions[r] {
            Source::Round(j) => prog.valid[j][p],
            _ => true,
        })).collect();
        assert!(!valid.is_empty(), "no plane valid in a batch of {planes}");
        assert!(valid.windows(2).all(|w| w[1] == w[0] + 1), "valid planes are one range: {valid:?}");
        assert!(valid.contains(&(planes / 2)));
    }
}
