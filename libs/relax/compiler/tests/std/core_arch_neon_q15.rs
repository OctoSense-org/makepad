// The 22 core::arch::aarch64 intrinsics moxcms 0.8's NEON paths call that the std here lacked
// ("cannot find function `vqrdmlah_s16` in this scope" and the like): 16-bit half-vector add, sub,
// min, max, dup, lane reads and loads, vmlal_s16, vqmovun_s32, the rounding doubling multiplies
// vqrdmulh_s16 / vqrdmulhq_s32 with their accumulating FEAT_RDM forms vqrdmlah / vqrdmlsh, and
// vcvtaq_u32_f32. Each runs over the same edge and pseudo-random inputs and its results are
// folded into one hash; the expected hashes are what the real instructions give (this file built
// with rustc on an Apple M-series CPU), so the emulation has to agree with the CPU bit for bit.
#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;
    use core::mem::transmute;

    const CASES: usize = 512 + 2048;
    const E16: [i16; 8] = [0, 1, -1, i16::MIN, i16::MAX, 0x4000, -0x4000, 0x7ffe];
    const E32: [i32; 8] = [0, 1, -1, i32::MIN, i32::MAX, 0x4000_0000, -0x4000_0000, 0x0001_8000];
    const EF: [f32; 16] = [
        0.0, -0.0, 0.5, -0.5, 1.5, 2.5, -1.5, 0.49999997, 8388609.0, 4294967040.0, 4294967296.0, 1.0e20, f32::INFINITY, f32::NEG_INFINITY,
        f32::NAN, 1.0e-40,
    ];

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }
    struct Hash(u64);
    impl Hash {
        fn bytes(&mut self, b: &[u8]) {
            for x in b {
                self.0 = (self.0 ^ *x as u64).wrapping_mul(0x100000001b3);
            }
        }
        fn i16s(&mut self, v: &[i16]) {
            for x in v {
                self.bytes(&x.to_le_bytes());
            }
        }
        fn i32s(&mut self, v: &[i32]) {
            for x in v {
                self.bytes(&x.to_le_bytes());
            }
        }
    }

    // operand `arg` of case `case`: the first 512 cases walk every triple of edge values through
    // lane 0 (the other lanes shifted along the table), the rest are random
    fn lanes16<const N: usize>(rng: &mut Rng, case: usize, arg: u32) -> [i16; N] {
        let mut r = [0i16; N];
        for (j, x) in r.iter_mut().enumerate() {
            *x = if case < 512 { E16[(case / 8usize.pow(arg) + j * (arg as usize + 1)) % 8] } else { rng.next() as i16 };
        }
        r
    }
    fn lanes32(rng: &mut Rng, case: usize, arg: u32) -> [i32; 4] {
        let mut r = [0i32; 4];
        for (j, x) in r.iter_mut().enumerate() {
            *x = if case < 512 { E32[(case / 8usize.pow(arg) + j * (arg as usize + 1)) % 8] } else { rng.next() as i32 };
        }
        r
    }
    unsafe fn h16(a: [i16; 4]) -> int16x4_t {
        transmute(a)
    }
    unsafe fn q16(a: [i16; 8]) -> int16x8_t {
        transmute(a)
    }
    unsafe fn q32(a: [i32; 4]) -> int32x4_t {
        transmute(a)
    }
    unsafe fn o16(v: int16x4_t) -> [i16; 4] {
        transmute(v)
    }
    unsafe fn ou16(v: uint16x4_t) -> [i16; 4] {
        transmute(v)
    }

    pub fn run() -> Vec<(&'static str, u64)> {
        let mut out = Vec::new();
        macro_rules! fold {
            ($name:literal, |$rng:ident, $case:ident, $h:ident| $body:block) => {{
                let mut $rng = Rng(0x9e3779b97f4a7c15);
                let mut $h = Hash(0xcbf29ce484222325);
                for $case in 0..CASES {
                    unsafe { $body }
                }
                out.push(($name, $h.0));
            }};
        }
        macro_rules! half_binary {
            ($($name:literal $f:ident $out:ident $cast:ty;)*) => {$(
                fold!($name, |rng, case, h| {
                    let (a, b) = (lanes16::<4>(&mut rng, case, 0), lanes16::<4>(&mut rng, case, 1));
                    let (a, b): ($cast, $cast) = (transmute(a), transmute(b));
                    h.i16s(&$out($f(a, b)));
                });
            )*};
        }
        half_binary! {
            "vadd_s16" vadd_s16 o16 int16x4_t;
            "vsub_s16" vsub_s16 o16 int16x4_t;
            "vmax_s16" vmax_s16 o16 int16x4_t;
            "vmin_s16" vmin_s16 o16 int16x4_t;
            "vmin_u16" vmin_u16 ou16 uint16x4_t;
            "vqrdmulh_s16" vqrdmulh_s16 o16 int16x4_t;
        }
        macro_rules! half_ternary {
            ($($name:literal $f:ident;)*) => {$(
                fold!($name, |rng, case, h| {
                    let (a, b, c) = (lanes16::<4>(&mut rng, case, 0), lanes16::<4>(&mut rng, case, 1), lanes16::<4>(&mut rng, case, 2));
                    h.i16s(&o16($f(h16(a), h16(b), h16(c))));
                });
            )*};
        }
        half_ternary! {
            "vqrdmlah_s16" vqrdmlah_s16;
            "vqrdmlsh_s16" vqrdmlsh_s16;
        }
        macro_rules! quad_ternary_16 {
            ($($name:literal $f:ident;)*) => {$(
                fold!($name, |rng, case, h| {
                    let (a, b, c) = (lanes16::<8>(&mut rng, case, 0), lanes16::<8>(&mut rng, case, 1), lanes16::<8>(&mut rng, case, 2));
                    let r: [i16; 8] = transmute($f(q16(a), q16(b), q16(c)));
                    h.i16s(&r);
                });
            )*};
        }
        quad_ternary_16! {
            "vqrdmlahq_s16" vqrdmlahq_s16;
            "vqrdmlshq_s16" vqrdmlshq_s16;
        }
        fold!("vqrdmlahq_s32", |rng, case, h| {
            let (a, b, c) = (lanes32(&mut rng, case, 0), lanes32(&mut rng, case, 1), lanes32(&mut rng, case, 2));
            let r: [i32; 4] = transmute(vqrdmlahq_s32(q32(a), q32(b), q32(c)));
            h.i32s(&r);
        });
        fold!("vqrdmulhq_s32", |rng, case, h| {
            let (a, b) = (lanes32(&mut rng, case, 0), lanes32(&mut rng, case, 1));
            let r: [i32; 4] = transmute(vqrdmulhq_s32(q32(a), q32(b)));
            h.i32s(&r);
        });
        fold!("vmlal_s16", |rng, case, h| {
            let (a, b, c) = (lanes32(&mut rng, case, 0), lanes16::<4>(&mut rng, case, 1), lanes16::<4>(&mut rng, case, 2));
            let r: [i32; 4] = transmute(vmlal_s16(q32(a), h16(b), h16(c)));
            h.i32s(&r);
        });
        fold!("vqmovun_s32", |rng, case, h| {
            let mut a = lanes32(&mut rng, case, 0);
            if case % 3 == 0 {
                // around the 16-bit boundaries, where the saturation starts
                a = [a[0] >> 15, a[1] >> 16, 0xffff + (a[2] >> 30), a[3] >> 14];
            }
            h.i16s(&ou16(vqmovun_s32(q32(a))));
        });
        fold!("vdup_n_s16", |rng, case, h| {
            let a = lanes16::<4>(&mut rng, case, 0);
            h.i16s(&o16(vdup_n_s16(a[case % 4])));
        });
        fold!("vdup_n_u16", |rng, case, h| {
            let a = lanes16::<4>(&mut rng, case, 0);
            h.i16s(&ou16(vdup_n_u16(a[case % 4] as u16)));
        });
        fold!("vget_lane_s16", |rng, case, h| {
            let v = h16(lanes16::<4>(&mut rng, case, 0));
            h.i16s(&[vget_lane_s16::<0>(v), vget_lane_s16::<1>(v), vget_lane_s16::<2>(v), vget_lane_s16::<3>(v)]);
        });
        fold!("vget_lane_u16", |rng, case, h| {
            let v: uint16x4_t = transmute(lanes16::<4>(&mut rng, case, 0));
            let r = [vget_lane_u16::<0>(v), vget_lane_u16::<1>(v), vget_lane_u16::<2>(v), vget_lane_u16::<3>(v)];
            h.i16s(&r.map(|x| x as i16));
        });
        // the loads read through a pointer of any alignment
        fold!("vld1_s16", |rng, case, h| {
            let src: [i16; 8] = lanes16::<8>(&mut rng, case, 0);
            let bytes: [u8; 16] = transmute(src);
            h.i16s(&o16(vld1_s16(bytes.as_ptr().add(case % 8).cast())));
        });
        fold!("vld1_dup_s16", |rng, case, h| {
            let src: [i16; 8] = lanes16::<8>(&mut rng, case, 0);
            let bytes: [u8; 16] = transmute(src);
            h.i16s(&o16(vld1_dup_s16(bytes.as_ptr().add(case % 15).cast())));
        });
        fold!("vld1q_dup_s32", |rng, case, h| {
            let src: [i32; 4] = lanes32(&mut rng, case, 0);
            let bytes: [u8; 16] = transmute(src);
            let r: [i32; 4] = transmute(vld1q_dup_s32(bytes.as_ptr().add(case % 13).cast()));
            h.i32s(&r);
        });
        fold!("vcvtaq_u32_f32", |rng, case, h| {
            let x: [f32; 4] = if case < 16 {
                [EF[case], -EF[case], EF[(case + 5) % 16], EF[(case + 11) % 16] * 0.5]
            } else if case < 1536 {
                // every quarter step around the ties
                let q = (case - 16) as f32 * 0.25;
                [q, -q, q + 4294965000.0, q * 2097152.0]
            } else {
                [f32::from_bits(rng.next() as u32), f32::from_bits(rng.next() as u32), (rng.next() % 100000) as f32 / 64.0, (rng.next() >> 40) as f32 / 256.0]
            };
            let v: float32x4_t = transmute(x);
            let r: [u32; 4] = transmute(vcvtaq_u32_f32(v));
            h.i32s(&r.map(|x| x as i32));
        });
        out
    }

    pub const EXPECTED: [(&str, u64); 22] = [
        ("vadd_s16", 0xf120c530d3b6ea8c),
        ("vsub_s16", 0xe44dbe52906efb63),
        ("vmax_s16", 0x55ad4c605188bf21),
        ("vmin_s16", 0xb15fe0ae19e783a5),
        ("vmin_u16", 0x5e22daf13e5478e8),
        ("vqrdmulh_s16", 0x8a7074384d6c07b9),
        ("vqrdmlah_s16", 0xadda9ba597643292),
        ("vqrdmlsh_s16", 0x4c6fb2e250a47adc),
        ("vqrdmlahq_s16", 0xacd0b40374898e0a),
        ("vqrdmlshq_s16", 0x0775c725fe3d9294),
        ("vqrdmlahq_s32", 0x7579d597c192468c),
        ("vqrdmulhq_s32", 0x55f44d861e801c48),
        ("vmlal_s16", 0x5aea8efdfaeab46f),
        ("vqmovun_s32", 0x2109e87dce11361d),
        ("vdup_n_s16", 0xbf23e39e9601fd05),
        ("vdup_n_u16", 0xbf23e39e9601fd05),
        ("vget_lane_s16", 0x1350ecabf36f2c3e),
        ("vget_lane_u16", 0x1350ecabf36f2c3e),
        ("vld1_s16", 0x4d76b0f1cb5a0b23),
        ("vld1_dup_s16", 0xeccc04455b4288bd),
        ("vld1q_dup_s32", 0xe913a673995a7e85),
        ("vcvtaq_u32_f32", 0x95400dd45422b126),
    ];
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_q15_intrinsics_match_the_real_instructions() {
    let got = neon::run();
    for ((name, hash), (_, want)) in got.iter().zip(neon::EXPECTED.iter()) {
        println!("| {:<16} {:016x} {}", name, hash, if hash == want { "ok" } else { "DIFFERS" });
    }
    assert_eq!(got.len(), 22);
    assert_eq!(got, neon::EXPECTED);
}

// a few values worked by hand, so a reader can see what the operations do
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_q15_worked_values() {
    use core::arch::aarch64::*;
    use core::mem::transmute;
    unsafe {
        let v = |a: [i16; 4]| -> int16x4_t { transmute(a) };
        let o = |v: int16x4_t| -> [i16; 4] { transmute(v) };
        // Q15: 0.5 * 0.5 = 0.25; MIN * MIN saturates; rounding goes up at the half
        assert_eq!(o(vqrdmulh_s16(v([0x4000, i16::MIN, 3, -3]), v([0x4000, i16::MIN, 0x4000, 0x4000]))), [0x2000, i16::MAX, 2, -1]);
        // accumulate: the sum is rounded and saturated once, so MIN * MIN (+1.0) on top of -1.0 is 0
        assert_eq!(o(vqrdmlah_s16(v([i16::MIN, 100, i16::MAX, -5]), v([i16::MIN, 0x4000, 0x4000, 0x2000]), v([i16::MIN, 0x4000, 0x4000, 4]))), [0, 100 + 0x2000, i16::MAX, -4]);
        assert_eq!(o(vqrdmlsh_s16(v([i16::MAX, 100, i16::MIN, 5]), v([i16::MIN, 0x4000, 0x4000, 0x2000]), v([i16::MIN, 0x4000, 0x4000, 4]))), [-1, 100 - 0x2000, i16::MIN, 4]);
        let w: [i32; 4] = transmute(vmlal_s16(transmute([1i32, -1, i32::MAX, 0]), v([2, -3, 1, i16::MIN]), v([5, 7, 1, i16::MIN])));
        assert_eq!(w, [11, -22, i32::MIN, 0x4000_0000]);
        let n: [u16; 4] = transmute(vqmovun_s32(transmute([-1i32, 0x1_0000, 0xffff, 12345])));
        assert_eq!(n, [0, 0xffff, 0xffff, 12345]);
        let c: [u32; 4] = transmute(vcvtaq_u32_f32(transmute([0.5f32, 2.5, -0.5, 1.0e20])));
        assert_eq!(c, [1, 3, 0, u32::MAX]);
        let d: [u16; 4] = transmute(vmin_u16(transmute([1u16, 0xffff, 7, 9]), vdup_n_u16(8)));
        assert_eq!(d, [1, 8, 7, 8]);
        assert_eq!(o(vmax_s16(v([1, -1, 7, 9]), vdup_n_s16(-1))), [1, -1, 7, 9]);
        assert_eq!(o(vsub_s16(vadd_s16(v([i16::MAX, 2, 3, 4]), vdup_n_s16(1)), vdup_n_s16(2))), [i16::MAX - 1, 1, 2, 3]);
        let src = [10i16, -20, 30, -40];
        assert_eq!(o(vld1_s16(src.as_ptr())), src);
        assert_eq!(o(vld1_dup_s16(src.as_ptr().add(1))), [-20; 4]);
        assert_eq!(vget_lane_s16::<3>(vld1_s16(src.as_ptr())), -40);
        assert_eq!(vget_lane_u16::<1>(transmute(src)), 65516);
        let e: [i32; 4] = transmute(vld1q_dup_s32([7i32].as_ptr()));
        assert_eq!(e, [7; 4]);
        let m: [i32; 4] = transmute(vqrdmulhq_s32(transmute([i32::MIN, 0x4000_0000, 3, -3]), transmute([i32::MIN, 0x4000_0000, 0x4000_0000, 0x4000_0000])));
        assert_eq!(m, [i32::MAX, 0x2000_0000, 2, -1]);
        let a: [i32; 4] = transmute(vqrdmlahq_s32(transmute([i32::MIN, 100, i32::MAX, -5]), transmute([i32::MIN, 0x4000_0000, 0x4000_0000, 2]), transmute([i32::MIN, 0x4000_0000, 0x4000_0000, 0x4000_0000])));
        assert_eq!(a, [0, 100 + 0x2000_0000, i32::MAX, -4]);
        let q: [i16; 8] = transmute(vqrdmlahq_s16(transmute([i16::MIN, 0, 0, 0, 1, 2, 3, 4]), transmute([i16::MIN; 8]), transmute([i16::MIN, 0, 1, -1, 0, 0, 0, 0])));
        assert_eq!(q, [0, 0, -1, 1, 1, 2, 3, 4]);
        let q: [i16; 8] = transmute(vqrdmlshq_s16(transmute([i16::MIN, 0, 0, 0, 1, 2, 3, 4]), transmute([i16::MIN; 8]), transmute([i16::MIN, 0, 1, -1, 0, 0, 0, 0])));
        assert_eq!(q, [i16::MIN, 0, 1, -1, 1, 2, 3, 4]);
    }
}
