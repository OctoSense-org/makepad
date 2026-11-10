// core::arch intrinsics lowered to vector instructions (libs/relax/compiler/src/simd.rs) against
// rustc's real ones: one line per intrinsic, "<name> <hash>" over xorshift inputs mixed with
// edge lanes (0, all ones, sign bits, 0x7f..). rustc -O --test and `makepad test` print the
// same lines (diff them); MAKEPAD_RUN_NO_SIMD=1 gives std's scalar versions.
// Each intrinsic runs twice: through a closure (inlined: vector values flow between
// intrinsics) and through a fn pointer (the intrinsic's own compiled body).

#![allow(unused_unsafe, non_snake_case, unused)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const N: usize = 512;

fn next(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// random bytes with edge lanes mixed in
fn rnd<T: Copy>(s: &mut u64) -> T {
    let mut buf = [0u64; 8];
    let mode = next(s) % 8;
    for w in buf.iter_mut() {
        let r = next(s);
        *w = match mode {
            0 => 0,
            1 => u64::MAX,
            2 => 0x8000_8000_8000_8000 ^ (r & 0x0101_0101_0101_0101),
            3 => 0x7f7f_7fff_7fff_ffff ^ (r & 0x0100_0001_0000_0080),
            4 => r & 0x00ff_00ff_00ff_00ff,
            _ => r,
        };
    }
    unsafe { core::ptr::read_unaligned(buf.as_ptr() as *const T) }
}

fn mix<T>(h: &mut u64, v: &T) {
    let b = unsafe { core::slice::from_raw_parts(v as *const T as *const u8, core::mem::size_of::<T>()) };
    for &x in b {
        *h = (*h ^ x as u64).wrapping_mul(0x100_0000_01b3);
    }
}

fn un<A: Copy, R, F: Fn(A) -> R>(name: &str, f: F, p: fn(A) -> R) {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut hp = h;
    for _ in 0..N {
        let a: A = rnd(&mut s);
        mix(&mut h, &f(a));
        mix(&mut hp, &p(core::hint::black_box(a)));
    }
    println!("{} {:016x} {}", name, h, if h == hp { "" } else { "PTR-DIFFERS" });
}

fn bin<A: Copy, B: Copy, R, F: Fn(A, B) -> R>(name: &str, f: F, p: fn(A, B) -> R) {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut hp = h;
    for _ in 0..N {
        let a: A = rnd(&mut s);
        let b: B = rnd(&mut s);
        mix(&mut h, &f(a, b));
        mix(&mut hp, &p(core::hint::black_box(a), b));
    }
    println!("{} {:016x} {}", name, h, if h == hp { "" } else { "PTR-DIFFERS" });
}

fn tri<A: Copy, B: Copy, C: Copy, R, F: Fn(A, B, C) -> R>(name: &str, f: F, p: fn(A, B, C) -> R) {
    let mut s = 0x1234_5678_9abc_def1u64;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut hp = h;
    for _ in 0..N {
        let a: A = rnd(&mut s);
        let b: B = rnd(&mut s);
        let c: C = rnd(&mut s);
        mix(&mut h, &f(a, b, c));
        mix(&mut hp, &p(core::hint::black_box(a), b, c));
    }
    println!("{} {:016x} {}", name, h, if h == hp { "" } else { "PTR-DIFFERS" });
}

/// A float lane's bits, for `fma`.
trait Lane: Copy + PartialEq {
    const MARKER: Self;
    fn nan(self) -> bool;
    fn quieted(self) -> Self;
}
impl Lane for u32 {
    const MARKER: u32 = 0x7fc0_0000;
    fn nan(self) -> bool {
        self & 0x7fff_ffff > 0x7f80_0000
    }
    fn quieted(self) -> u32 {
        self | 0x0040_0000
    }
}
impl Lane for u64 {
    const MARKER: u64 = 0x7ff8_0000_0000_0000;
    fn nan(self) -> bool {
        self & 0x7fff_ffff_ffff_ffff > 0x7ff0_0000_0000_0000
    }
    fn quieted(self) -> u64 {
        self | 0x0008_0000_0000_0000
    }
}

/// `tri` for a packed fused multiply-add of L lanes of E's bits. A lane with two or more NaN
/// operands may return any of them: llvm.fma leaves the choice open and rustc's pick depends on
/// register allocation (b's NaN at 128 bits, a's at 256). Such a lane hashes as one marker when its
/// result is one of those NaNs, quieted; any other result still shows.
fn fma<V: Copy, E: Lane, const L: usize>(name: &str, f: impl Fn(V, V, V) -> V, p: fn(V, V, V) -> V) {
    let lanes = |v: V| -> [E; L] { unsafe { core::ptr::read_unaligned(&v as *const V as *const [E; L]) } };
    let masked = |r: V, a: V, b: V, c: V| -> [E; L] {
        let (mut r, a, b, c) = (lanes(r), lanes(a), lanes(b), lanes(c));
        for l in 0..L {
            let ops = [a[l], b[l], c[l]];
            let nans = ops.iter().filter(|x| x.nan());
            if nans.clone().count() >= 2 && nans.clone().any(|x| x.quieted() == r[l]) {
                r[l] = E::MARKER;
            }
        }
        r
    };
    let mut s = 0x1234_5678_9abc_def1u64;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut hp = h;
    for _ in 0..N {
        let a: V = rnd(&mut s);
        let b: V = rnd(&mut s);
        let c: V = rnd(&mut s);
        mix(&mut h, &masked(f(a, b, c), a, b, c));
        mix(&mut hp, &masked(p(core::hint::black_box(a), b, c), a, b, c));
    }
    println!("{} {:016x} {}", name, h, if h == hp { "" } else { "PTR-DIFFERS" });
}

macro_rules! u {
    ($name:ident, $a:ty) => {
        un(stringify!($name), |a: $a| unsafe { $name(a) }, |a: $a| unsafe { $name(a) })
    };
    ($name:ident :: <$k:literal>, $a:ty) => {
        un(concat!(stringify!($name), "::<", $k, ">"), |a: $a| unsafe { $name::<$k>(a) }, |a: $a| unsafe { $name::<$k>(a) })
    };
}
macro_rules! b {
    ($name:ident, $a:ty, $b:ty) => {
        bin(stringify!($name), |a: $a, b: $b| unsafe { $name(a, b) }, |a: $a, b: $b| unsafe { $name(a, b) })
    };
    ($name:ident :: <$k:literal>, $a:ty, $b:ty) => {
        bin(concat!(stringify!($name), "::<", $k, ">"), |a: $a, b: $b| unsafe { $name::<$k>(a, b) }, |a: $a, b: $b| unsafe { $name::<$k>(a, b) })
    };
}
/// rustc's legacy const-generic call form (`vshrq_n_u8(a, 4)`): the constant passed as the last
/// argument of an intrinsic marked `#[rustc_legacy_const_generics]`
macro_rules! ul {
    ($name:ident ($k:literal), $a:ty) => {
        un(concat!(stringify!($name), "::legacy::", $k), |a: $a| unsafe { $name(a, $k) }, |a: $a| unsafe { $name(a, $k) })
    };
}
macro_rules! bl {
    ($name:ident ($k:literal), $a:ty, $b:ty) => {
        bin(concat!(stringify!($name), "::legacy::", $k), |a: $a, b: $b| unsafe { $name(a, b, $k) }, |a: $a, b: $b| unsafe { $name(a, b, $k) })
    };
}
macro_rules! t {
    ($name:ident, $a:ty, $b:ty, $c:ty) => {
        tri(stringify!($name), |a: $a, b: $b, c: $c| unsafe { $name(a, b, c) }, |a: $a, b: $b, c: $c| unsafe { $name(a, b, c) })
    };
}
macro_rules! fma {
    ($name:ident, $v:ty, $e:ty, $l:literal) => {
        fma::<$v, $e, $l>(stringify!($name), |a: $v, b: $v, c: $v| unsafe { $name(a, b, c) }, |a: $v, b: $v, c: $v| unsafe { $name(a, b, c) })
    };
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn chain_adler(a: [u8; 64], b: [u8; 64]) -> [u32; 4] {
    unsafe {
            let mut s1 = vdupq_n_u32(1);
            let mut s2 = vdupq_n_u32(0);
            let mut w = vdupq_n_u16(0);
            let mut i = 0;
            while i < 64 {
                let x = vld1q_u8(a.as_ptr().add(i));
                let y = vld1q_u8(b.as_ptr().add(i));
                s2 = vaddq_u32(s2, vshlq_n_u32::<4>(s1));
                w = vpadalq_u8(w, x);
                w = vaddw_u8(w, vget_low_u8(y));
                s1 = vpadalq_u16(s1, vpaddlq_u8(x));
                s2 = vmlal_high_u16(s2, w, vpaddlq_u8(y));
                s2 = vmlal_u16(s2, vget_low_u16(w), vget_high_u16(w));
                i += 16;
            }
            [vaddvq_u32(s1), vaddvq_u32(s2), vmaxvq_u32(s2), vgetq_lane_u32::<1>(s1)]
        }
}

/// bytecount 0.6's NEON count: four registers loaded at once and destructured, lanes compared
/// with a needle and summed across with UADDLV.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn chain_count(a: [u8; 64], b: [u8; 64]) -> [u16; 4] {
    unsafe {
        let mask = vdupq_n_u8(0x81);
        let needles = vdupq_n_u8(b[0] & 0x81);
        let uint8x16x4_t(h1, h2, h3, h4) = vld1q_u8_x4(a.as_ptr());
        let mut count = vdupq_n_u8(0);
        for h in [h1, h2, h3, h4] {
            count = vsubq_u8(count, vceqq_u8(vandq_u8(h, mask), needles));
        }
        [vaddlvq_u8(count), vaddlvq_u8(h1), vaddlvq_u8(vaddq_u8(h2, h3)), vaddlvq_u8(h4)]
    }
}

/// aho-corasick 1.1's NEON Teddy step: intrinsics called in the legacy const-generic form, one
/// forwarding the caller's own const parameter (`vshrq_n_u8(self, BITS)`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn shift_8bit_lane_right<const BITS: i32>(v: uint8x16_t) -> uint8x16_t {
    vshrq_n_u8(v, BITS)
}

/// One Teddy chunk: nibble lookups, then EXT shifting the previous chunk's last bytes in
/// (`vextq_u8(prev, res, 15/14/13)`) and the is_zero test over 64-bit lanes.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn chain_teddy(a: [u8; 32], b: [u8; 32]) -> [u8; 72] {
    unsafe {
        let prev = vld1q_u8(a.as_ptr());
        let cur = vld1q_u8(a.as_ptr().add(16));
        let lo = vqtbl1q_u8(vld1q_u8(b.as_ptr()), vandq_u8(cur, vdupq_n_u8(0xf)));
        let hi = vqtbl1q_u8(vld1q_u8(b.as_ptr().add(16)), shift_8bit_lane_right::<4>(cur));
        let res = vandq_u8(lo, hi);
        let mut out = [0u8; 72];
        for (k, v) in [res, vextq_u8(prev, res, 15), vextq_u8(prev, res, 14), vextq_u8(prev, res, 13)].into_iter().enumerate() {
            vst1q_u8(out.as_mut_ptr().add(16 * k), v);
        }
        for (k, v) in [res, vandq_u8(res, vdupq_n_u8(0)), vceqq_u8(prev, prev), vorrq_u8(lo, vshlq_n_u8(hi, 7))].into_iter().enumerate() {
            let maxes = vreinterpretq_u64_u8(vpmaxq_u8(v, v));
            out[64 + 2 * k] = (vgetq_lane_u64(maxes, 0) == 0) as u8;
            out[65 + 2 * k] = (vgetq_lane_u64(maxes, 1) == vgetq_lane_u64::<0>(maxes)) as u8;
        }
        out
    }
}

#[cfg(target_arch = "aarch64")]
fn arch() {
    tri("f32::mul_add", |a: f32, b: f32, c: f32| a.mul_add(b, c), |a: f32, b: f32, c: f32| a.mul_add(b, c));
    tri("f64::mul_add", |a: f64, b: f64, c: f64| a.mul_add(b, c), |a: f64, b: f64, c: f64| a.mul_add(b, c));
    b!(vaddq_s16, int16x8_t, int16x8_t);
    b!(vaddq_s32, int32x4_t, int32x4_t);
    b!(vaddq_u32, uint32x4_t, uint32x4_t);
    b!(vsubq_s16, int16x8_t, int16x8_t);
    b!(vsubq_s32, int32x4_t, int32x4_t);
    b!(vmulq_s16, int16x8_t, int16x8_t);
    b!(vmulq_s32, int32x4_t, int32x4_t);
    b!(vmaxq_s16, int16x8_t, int16x8_t);
    b!(vminq_s16, int16x8_t, int16x8_t);
    b!(vorrq_s32, int32x4_t, int32x4_t);
    b!(vzip1q_s16, int16x8_t, int16x8_t);
    b!(vzip2q_s16, int16x8_t, int16x8_t);
    b!(vtrn1q_s64, int64x2_t, int64x2_t);
    b!(vtrn2q_s64, int64x2_t, int64x2_t);
    u!(vdupq_n_u8, u8);
    u!(vdupq_n_s16, i16);
    u!(vdupq_n_u16, u16);
    u!(vdupq_n_s32, i32);
    u!(vdupq_n_u32, u32);
    u!(vgetq_lane_u32::<0>, uint32x4_t);
    u!(vgetq_lane_u32::<3>, uint32x4_t);
    u!(vget_lane_u32::<0>, uint32x2_t);
    u!(vget_lane_u32::<1>, uint32x2_t);
    b!(vsetq_lane_u32::<0>, u32, uint32x4_t);
    b!(vsetq_lane_u32::<2>, u32, uint32x4_t);
    u!(vaddvq_u32, uint32x4_t);
    u!(vaddlvq_u8, uint8x16_t);
    u!(vmaxvq_u32, uint32x4_t);
    b!(vaddw_u8, uint16x8_t, uint8x8_t);
    t!(vmlal_u16, uint32x4_t, uint16x4_t, uint16x4_t);
    t!(vmlal_high_u16, uint32x4_t, uint16x8_t, uint16x8_t);
    b!(vpadalq_u8, uint16x8_t, uint8x16_t);
    b!(vpadalq_u16, uint32x4_t, uint16x8_t);
    u!(vpaddlq_u8, uint8x16_t);
    u!(vshlq_n_u32::<0>, uint32x4_t);
    u!(vshlq_n_u32::<5>, uint32x4_t);
    u!(vshlq_n_u32::<31>, uint32x4_t);
    u!(vqshlq_n_u32::<0>, uint32x4_t);
    u!(vqshlq_n_u32::<6>, uint32x4_t);
    u!(vqshlq_n_u32::<31>, uint32x4_t);
    u!(vshrq_n_s16::<1>, int16x8_t);
    u!(vshrq_n_s16::<16>, int16x8_t);
    u!(vshrq_n_s32::<7>, int32x4_t);
    u!(vshrq_n_s32::<32>, int32x4_t);
    // every immediate shift in the legacy form rustc accepts (stdarch's legacy position 1)
    ul!(vshlq_n_s8(0), int8x16_t);
    ul!(vshlq_n_s8(7), int8x16_t);
    ul!(vshrq_n_s8(1), int8x16_t);
    ul!(vshrq_n_s8(8), int8x16_t);
    ul!(vshlq_n_s16(15), int16x8_t);
    ul!(vshrq_n_s16(3), int16x8_t);
    ul!(vshlq_n_s32(9), int32x4_t);
    ul!(vshrq_n_s32(32), int32x4_t);
    ul!(vshlq_n_s64(63), int64x2_t);
    ul!(vshrq_n_s64(64), int64x2_t);
    ul!(vshlq_n_u8(3), uint8x16_t);
    ul!(vshrq_n_u8(4), uint8x16_t);
    ul!(vshrq_n_u8(8), uint8x16_t);
    ul!(vshlq_n_u16(1), uint16x8_t);
    ul!(vshrq_n_u16(16), uint16x8_t);
    ul!(vshlq_n_u32(31), uint32x4_t);
    ul!(vshrq_n_u32(5), uint32x4_t);
    ul!(vshlq_n_u64(32), uint64x2_t);
    ul!(vshrq_n_u64(1), uint64x2_t);
    ul!(vgetq_lane_u64(1), uint64x2_t);
    ul!(vgetq_lane_u8(0), uint8x16_t);
    ul!(vgetq_lane_u8(15), uint8x16_t);
    u!(vmvnq_u8, uint8x16_t);
    // vreinterpret: zmij's and chacha20's, then one per element type and width (p128 = u128)
    u!(vreinterpret_s32_u64, uint64x1_t);
    u!(vreinterpret_u32_s32, int32x2_t);
    u!(vreinterpret_u16_s32, int32x2_t);
    u!(vreinterpretq_u64_u32, uint32x4_t);
    u!(vreinterpretq_u32_u64, uint64x2_t);
    u!(vreinterpret_f64_s8, int8x8_t);
    u!(vreinterpret_p16_f32, float32x2_t);
    u!(vreinterpret_p64_u16, uint16x4_t);
    u!(vreinterpret_s64_p8, poly8x8_t);
    u!(vreinterpretq_s8_f64, float64x2_t);
    u!(vreinterpretq_p8_s16, int16x8_t);
    u!(vreinterpretq_p64_s32, int32x4_t);
    u!(vreinterpretq_f32_p16, poly16x8_t);
    u!(vreinterpretq_p128_u8, uint8x16_t);
    u!(vreinterpretq_u16_p128, u128);
    // zmij's digit split and chacha20's rotate
    t!(vmla_n_s32, int32x2_t, int32x2_t, i32);
    t!(vmlaq_n_s16, int16x8_t, int16x8_t, i16);
    t!(vmlaq_n_s32, int32x4_t, int32x4_t, i32);
    b!(vqdmulh_n_s32, int32x2_t, i32);
    b!(vqdmulhq_n_s16, int16x8_t, i16);
    b!(vqdmulhq_n_s32, int32x4_t, i32);
    u!(vrev64q_u8, uint8x16_t);
    u!(vrev32q_u16, uint16x8_t);
    ul!(vshll_n_u16(0), uint16x4_t);
    ul!(vshll_n_u16(9), uint16x4_t);
    ul!(vshll_n_u16(16), uint16x4_t);
    ul!(vshr_n_u32(1), uint32x2_t);
    ul!(vshr_n_u32(9), uint32x2_t);
    ul!(vshr_n_u32(32), uint32x2_t);
    // memchr 2.8, simdutf8 0.1.5, zmij 1.0.19+
    u!(vrev16q_u8, uint8x16_t);
    u!(vmovq_n_u8, u8);
    u!(vcgtzq_s8, int8x16_t);
    b!(vcombine_s32, int32x2_t, int32x2_t);
    u!(vcreate_u64, u64);
    u!(vdup_n_s32, i32);
    // sha2's SHA-512 backend
    b!(vextq_u64::<0>, uint64x2_t, uint64x2_t);
    bl!(vextq_u64(1), uint64x2_t, uint64x2_t);
    t!(vsha512hq_u64, uint64x2_t, uint64x2_t, uint64x2_t);
    t!(vsha512h2q_u64, uint64x2_t, uint64x2_t, uint64x2_t);
    b!(vsha512su0q_u64, uint64x2_t, uint64x2_t);
    t!(vsha512su1q_u64, uint64x2_t, uint64x2_t, uint64x2_t);
    // the aes crate's aarch64 backend
    b!(vaeseq_u8, uint8x16_t, uint8x16_t);
    b!(vaesdq_u8, uint8x16_t, uint8x16_t);
    u!(vaesmcq_u8, uint8x16_t);
    u!(vaesimcq_u8, uint8x16_t);
    // one full round each way, as aes chains them
    bin("chain_aes", |a: uint8x16_t, k: uint8x16_t| unsafe { [vaesmcq_u8(vaeseq_u8(a, k)), vaesimcq_u8(vaesdq_u8(a, k))] }, |a: uint8x16_t, k: uint8x16_t| unsafe {
        [vaesmcq_u8(vaeseq_u8(a, k)), vaesimcq_u8(vaesdq_u8(a, k))]
    });
    u!(vshrq_n_u8::<1>, uint8x16_t);
    u!(vshlq_n_u8::<7>, uint8x16_t);
    b!(vextq_u8::<0>, uint8x16_t, uint8x16_t);
    b!(vextq_u8::<1>, uint8x16_t, uint8x16_t);
    bl!(vextq_u8(13), uint8x16_t, uint8x16_t);
    bl!(vextq_u8(15), uint8x16_t, uint8x16_t);
    bl!(vextq_u32(3), uint32x4_t, uint32x4_t);
    u!(vget_low_u8, uint8x16_t);
    u!(vget_low_u16, uint16x8_t);
    u!(vget_low_u32, uint32x4_t);
    u!(vget_low_s16, int16x8_t);
    u!(vget_high_u8, uint8x16_t);
    u!(vget_high_u16, uint16x8_t);
    u!(vget_high_u32, uint32x4_t);
    b!(vcombine_u8, uint8x8_t, uint8x8_t);
    b!(vceq_u8, uint8x8_t, uint8x8_t);
    u!(vcltz_s8, int8x8_t);
    u!(vdup_n_u8, u8);
    u!(vreinterpret_u64_u8, uint8x8_t);
    u!(vget_lane_u64::<0>, uint64x1_t);
    un("vld1_u8", |p: [u8; 9]| unsafe { vld1_u8(p.as_ptr().add(1)) }, |p: [u8; 9]| unsafe { vld1_u8(p.as_ptr().add(1)) });
    b!(vcombine_u16, uint16x4_t, uint16x4_t);
    b!(vcombine_s16, int16x4_t, int16x4_t);
    un("vld1q_u8", |p: [u8; 17]| unsafe { vld1q_u8(p.as_ptr().add(1)) }, |p: [u8; 17]| unsafe { vld1q_u8(p.as_ptr().add(1)) });
    un(
        "vld1q_u8_x4",
        |p: [u8; 64]| {
            let mut b = [0u8; 65];
            b[1..].copy_from_slice(&p);
            unsafe { vld1q_u8_x4(b.as_ptr().add(1)) }
        },
        |p: [u8; 64]| {
            let mut b = [0u8; 65];
            b[1..].copy_from_slice(&p);
            unsafe { vld1q_u8_x4(b.as_ptr().add(1)) }
        },
    );
    un("vld1q_u16", |p: [u16; 8]| unsafe { vld1q_u16(p.as_ptr()) }, |p: [u16; 8]| unsafe { vld1q_u16(p.as_ptr()) });
    un("vld1_u16", |p: [u16; 4]| unsafe { vld1_u16(p.as_ptr()) }, |p: [u16; 4]| unsafe { vld1_u16(p.as_ptr()) });
    un("vld1q_u32", |p: [u32; 4]| unsafe { vld1q_u32(p.as_ptr()) }, |p: [u32; 4]| unsafe { vld1q_u32(p.as_ptr()) });
    un("vld1q_s16", |p: [i16; 8]| unsafe { vld1q_s16(p.as_ptr()) }, |p: [i16; 8]| unsafe { vld1q_s16(p.as_ptr()) });
    un(
        "vst1q_u32",
        |a: uint32x4_t| {
            let mut o = [0u32; 6];
            unsafe { vst1q_u32(o.as_mut_ptr().add(1), a) };
            o
        },
        |a: uint32x4_t| {
            let mut o = [0u32; 6];
            unsafe { vst1q_u32(o.as_mut_ptr().add(1), a) };
            o
        },
    );
    un(
        "vst1q_s16",
        |a: int16x8_t| {
            let mut o = [0i16; 10];
            unsafe { vst1q_s16(o.as_mut_ptr().add(1), a) };
            o
        },
        |a: int16x8_t| {
            let mut o = [0i16; 10];
            unsafe { vst1q_s16(o.as_mut_ptr().add(1), a) };
            o
        },
    );
    // a chain in a loop: adler32-style sums (vector values across iterations)
    bin(
        "chain_adler",
        |a: [u8; 64], b: [u8; 64]| chain_adler(a, b),
        |a: [u8; 64], b: [u8; 64]| {
            let mut h = 0u32;
            for i in 0..64 {
                h = h.wrapping_mul(31).wrapping_add(a[i] as u32 ^ b[i] as u32);
            }
            [h, 0, 0, 0]
        },
    );
    bin("chain_teddy", |a: [u8; 32], b: [u8; 32]| chain_teddy(a, b), |a: [u8; 32], b: [u8; 32]| {
        let res: [u8; 16] = core::array::from_fn(|i| b[(a[16 + i] & 0xf) as usize] & b[16 + (a[16 + i] >> 4) as usize]);
        let ext = |k: usize| -> [u8; 16] { core::array::from_fn(|i| if i + k < 16 { a[i + k] } else { res[i + k - 16] }) };
        let mut out = [0u8; 72];
        for (k, v) in [res, ext(15), ext(14), ext(13)].into_iter().enumerate() {
            out[16 * k..16 * k + 16].copy_from_slice(&v);
        }
        let lo: [u8; 16] = core::array::from_fn(|i| b[(a[16 + i] & 0xf) as usize]);
        let hi: [u8; 16] = core::array::from_fn(|i| b[16 + (a[16 + i] >> 4) as usize]);
        let tests = [res, [0; 16], [0xff; 16], core::array::from_fn(|i| lo[i] | hi[i] << 7)];
        for (k, v) in tests.into_iter().enumerate() {
            // vpmaxq_u8(v, v): the pairwise maxima of v, twice; both 64-bit lanes hold the same bytes
            let zero = (0..8).all(|j| v[2 * j].max(v[2 * j + 1]) == 0);
            out[64 + 2 * k] = zero as u8;
            out[65 + 2 * k] = 1;
        }
        out
    });
    // the same in scalar code on the pointer side: PTR-DIFFERS when the NEON chain is wrong
    bin(
        "chain_count",
        |a: [u8; 64], b: [u8; 64]| chain_count(a, b),
        |a: [u8; 64], b: [u8; 64]| {
            let sum = |s: &[u8]| s.iter().map(|&x| x as u16).sum::<u16>();
            let count = a.iter().filter(|&&x| x & 0x81 == b[0] & 0x81).count() as u16;
            let middle = (0..16).map(|i| a[16 + i].wrapping_add(a[32 + i]) as u16).sum::<u16>();
            [count, sum(&a[..16]), middle, sum(&a[48..])]
        },
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn chain_adler(a: [u8; 64], b: [u8; 64]) -> [i32; 4] {
    unsafe {
            let ones = _mm256_set1_epi16(1);
            let mut s1 = _mm256_setzero_si256();
            let mut s2 = _mm256_setzero_si256();
            let mut i = 0;
            while i < 64 {
                let x = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                let y = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
                s2 = _mm256_add_epi32(s2, _mm256_slli_epi32::<5>(s1));
                s1 = _mm256_add_epi32(s1, _mm256_sad_epu8(x, _mm256_setzero_si256()));
                let m = _mm256_maddubs_epi16(y, x);
                s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(m, ones));
                i += 32;
            }
            let lo = _mm256_castsi256_si128(s2);
            let hi = _mm256_extracti128_si256::<1>(s2);
            let t = _mm_add_epi32(lo, hi);
            let t = _mm_add_epi32(t, _mm_shuffle_epi32::<0x4e>(t));
            let t = _mm_add_epi32(t, _mm_srli_si128::<4>(t));
            [_mm_cvtsi128_si32(t), _mm_extract_epi32::<2>(_mm256_castsi256_si128(s1)), 0, 0]
        }
}

#[cfg(target_arch = "x86_64")]
fn arch() {
    b!(_mm_add_epi8, __m128i, __m128i);
    b!(_mm_add_epi16, __m128i, __m128i);
    b!(_mm_add_epi32, __m128i, __m128i);
    b!(_mm_sub_epi8, __m128i, __m128i);
    b!(_mm_sub_epi16, __m128i, __m128i);
    b!(_mm_and_si128, __m128i, __m128i);
    b!(_mm_or_si128, __m128i, __m128i);
    b!(_mm_xor_si128, __m128i, __m128i);
    b!(_mm_andnot_si128, __m128i, __m128i);
    b!(_mm_avg_epu8, __m128i, __m128i);
    b!(_mm_cmpeq_epi16, __m128i, __m128i);
    b!(_mm_cmpeq_epi8, __m128i, __m128i);
    b!(_mm_cmpgt_epi8, __m128i, __m128i);
    u!(_mm_movemask_epi8, __m128i);
    un(
        "_mm_store_si128",
        |a: __m128i| {
            let mut o = [a, a];
            unsafe { _mm_store_si128(&mut o[1], _mm_setzero_si128()) };
            o
        },
        |a: __m128i| {
            let mut o = [a, a];
            unsafe { _mm_store_si128(&mut o[1], _mm_setzero_si128()) };
            o
        },
    );
    b!(_mm_min_epi16, __m128i, __m128i);
    b!(_mm_madd_epi16, __m128i, __m128i);
    b!(_mm_maddubs_epi16, __m128i, __m128i);
    b!(_mm_sad_epu8, __m128i, __m128i);
    b!(_mm_packus_epi16, __m128i, __m128i);
    b!(_mm_unpacklo_epi8, __m128i, __m128i);
    b!(_mm_unpackhi_epi8, __m128i, __m128i);
    b!(_mm_unpackhi_epi64, __m128i, __m128i);
    b!(_mm_shuffle_epi8, __m128i, __m128i);
    u!(_mm_abs_epi16, __m128i);
    u!(_mm_shuffle_epi32::<0>, __m128i);
    u!(_mm_shuffle_epi32::<27>, __m128i);
    u!(_mm_shuffle_epi32::<177>, __m128i);
    u!(_mm_shuffle_epi32::<255>, __m128i);
    u!(_mm_slli_epi32::<0>, __m128i);
    u!(_mm_slli_epi32::<9>, __m128i);
    u!(_mm_slli_epi32::<31>, __m128i);
    u!(_mm_slli_epi32::<32>, __m128i);
    u!(_mm_srli_si128::<0>, __m128i);
    u!(_mm_srli_si128::<4>, __m128i);
    u!(_mm_srli_si128::<8>, __m128i);
    u!(_mm_srli_si128::<15>, __m128i);
    u!(_mm_srli_si128::<16>, __m128i);
    u!(_mm_set1_epi8, i8);
    u!(_mm_set1_epi16, i16);
    un("_mm_setzero_si128", |_a: u8| unsafe { _mm_setzero_si128() }, |_a: u8| unsafe { _mm_setzero_si128() });
    b!(_mm_set_epi64x, i64, i64);
    u!(_mm_cvtsi32_si128, i32);
    u!(_mm_cvtsi128_si32, __m128i);
    u!(_mm_extract_epi32::<0>, __m128i);
    u!(_mm_extract_epi32::<3>, __m128i);
    un("_mm_loadu_si128", |p: [u8; 17]| unsafe { _mm_loadu_si128(p.as_ptr().add(1) as *const __m128i) }, |p: [u8; 17]| unsafe { _mm_loadu_si128(p.as_ptr().add(1) as *const __m128i) });
    un("_mm_load_si128", |p: __m128i| unsafe { _mm_load_si128(&p) }, |p: __m128i| unsafe { _mm_load_si128(&p) });
    un(
        "_mm_storeu_si128",
        |a: __m128i| {
            let mut o = [0u8; 20];
            unsafe { _mm_storeu_si128(o.as_mut_ptr().add(3) as *mut __m128i, a) };
            o
        },
        |a: __m128i| {
            let mut o = [0u8; 20];
            unsafe { _mm_storeu_si128(o.as_mut_ptr().add(3) as *mut __m128i, a) };
            o
        },
    );
    b!(_mm256_add_epi16, __m256i, __m256i);
    b!(_mm256_add_epi32, __m256i, __m256i);
    b!(_mm256_sub_epi16, __m256i, __m256i);
    b!(_mm256_sub_epi32, __m256i, __m256i);
    b!(_mm256_mullo_epi16, __m256i, __m256i);
    b!(_mm256_mullo_epi32, __m256i, __m256i);
    b!(_mm256_max_epi16, __m256i, __m256i);
    b!(_mm256_min_epi16, __m256i, __m256i);
    b!(_mm256_or_si256, __m256i, __m256i);
    b!(_mm256_madd_epi16, __m256i, __m256i);
    b!(_mm256_maddubs_epi16, __m256i, __m256i);
    b!(_mm256_sad_epu8, __m256i, __m256i);
    b!(_mm256_packus_epi16, __m256i, __m256i);
    b!(_mm256_packus_epi32, __m256i, __m256i);
    b!(_mm256_packs_epi32, __m256i, __m256i);
    b!(_mm256_shuffle_epi8, __m256i, __m256i);
    b!(_mm256_unpacklo_epi8, __m256i, __m256i);
    b!(_mm256_unpackhi_epi8, __m256i, __m256i);
    b!(_mm256_unpacklo_epi16, __m256i, __m256i);
    b!(_mm256_unpackhi_epi16, __m256i, __m256i);
    b!(_mm256_unpacklo_epi32, __m256i, __m256i);
    b!(_mm256_unpackhi_epi32, __m256i, __m256i);
    b!(_mm256_unpacklo_epi64, __m256i, __m256i);
    b!(_mm256_unpackhi_epi64, __m256i, __m256i);
    // 256-bit and fused float operations (two 128-bit halves in Relax)
    b!(_mm_shuffle_ps::<0>, __m128, __m128);
    b!(_mm_shuffle_ps::<0x1b>, __m128, __m128);
    b!(_mm_shuffle_ps::<0xb1>, __m128, __m128);
    b!(_mm_shuffle_ps::<0x4e>, __m128, __m128);
    b!(_mm_shuffle_ps::<0xff>, __m128, __m128);
    b!(_mm_movehl_ps, __m128, __m128);
    b!(_mm_unpacklo_ps, __m128, __m128);
    b!(_mm_unpackhi_ps, __m128, __m128);
    b!(_mm256_add_ps, __m256, __m256);
    b!(_mm256_sub_ps, __m256, __m256);
    b!(_mm256_mul_ps, __m256, __m256);
    b!(_mm256_div_ps, __m256, __m256);
    b!(_mm256_max_ps, __m256, __m256);
    b!(_mm256_min_ps, __m256, __m256);
    b!(_mm256_and_ps, __m256, __m256);
    b!(_mm256_or_ps, __m256, __m256);
    b!(_mm256_xor_ps, __m256, __m256);
    b!(_mm256_andnot_ps, __m256, __m256);
    fma!(_mm256_fmadd_ps, __m256, u32, 8);
    fma!(_mm256_fmsub_ps, __m256, u32, 8);
    fma!(_mm256_fnmadd_ps, __m256, u32, 8);
    fma!(_mm256_fnmsub_ps, __m256, u32, 8);
    fma!(_mm_fmadd_ps, __m128, u32, 4);
    fma!(_mm_fmsub_ps, __m128, u32, 4);
    fma!(_mm_fnmadd_ps, __m128, u32, 4);
    fma!(_mm_fnmsub_ps, __m128, u32, 4);
    fma!(_mm_fmadd_pd, __m128d, u64, 2);
    fma!(_mm_fmsub_pd, __m128d, u64, 2);
    fma!(_mm_fnmadd_pd, __m128d, u64, 2);
    fma!(_mm_fnmsub_pd, __m128d, u64, 2);
    fma!(_mm_fmaddsub_pd, __m128d, u64, 2);
    fma!(_mm_fmsubadd_pd, __m128d, u64, 2);
    fma!(_mm_fmadd_sd, __m128d, u64, 2);
    fma!(_mm_fmsub_sd, __m128d, u64, 2);
    fma!(_mm_fnmadd_sd, __m128d, u64, 2);
    fma!(_mm_fnmsub_sd, __m128d, u64, 2);
    fma!(_mm256_fmadd_pd, __m256d, u64, 4);
    fma!(_mm256_fmsub_pd, __m256d, u64, 4);
    fma!(_mm256_fnmadd_pd, __m256d, u64, 4);
    fma!(_mm256_fnmsub_pd, __m256d, u64, 4);
    fma!(_mm256_fmaddsub_pd, __m256d, u64, 4);
    fma!(_mm256_fmsubadd_pd, __m256d, u64, 4);
    b!(_mm256_and_si256, __m256i, __m256i);
    b!(_mm256_cmpeq_epi8, __m256i, __m256i);
    u!(_mm256_set1_ps, f32);
    un("_mm256_setzero_ps", |_a: u8| unsafe { _mm256_setzero_ps() }, |_a: u8| unsafe { _mm256_setzero_ps() });
    u!(_mm256_castps256_ps128, __m256);
    u!(_mm256_extractf128_ps::<0>, __m256);
    u!(_mm256_extractf128_ps::<1>, __m256);
    un("_mm256_loadu_ps", |p: [u8; 40]| unsafe { _mm256_loadu_ps(p.as_ptr().add(5) as *const f32) }, |p: [u8; 40]| unsafe { _mm256_loadu_ps(p.as_ptr().add(5) as *const f32) });
    un(
        "_mm256_storeu_ps",
        |a: __m256| {
            let mut o = [0u8; 40];
            unsafe { _mm256_storeu_ps(o.as_mut_ptr().add(5) as *mut f32, a) };
            o
        },
        |a: __m256| {
            let mut o = [0u8; 40];
            unsafe { _mm256_storeu_ps(o.as_mut_ptr().add(5) as *mut f32, a) };
            o
        },
    );
    // scalar fused multiply-add (f32::mul_add / f64::mul_add)
    tri("f32::mul_add", |a: f32, b: f32, c: f32| a.mul_add(b, c), |a: f32, b: f32, c: f32| a.mul_add(b, c));
    tri("f64::mul_add", |a: f64, b: f64, c: f64| a.mul_add(b, c), |a: f64, b: f64, c: f64| a.mul_add(b, c));
    u!(_mm256_set1_epi16, i16);
    u!(_mm256_set1_epi32, i32);
    un("_mm256_setzero_si256", |_a: u8| unsafe { _mm256_setzero_si256() }, |_a: u8| unsafe { _mm256_setzero_si256() });
    u!(_mm256_slli_epi16::<0>, __m256i);
    u!(_mm256_slli_epi16::<3>, __m256i);
    u!(_mm256_slli_epi16::<16>, __m256i);
    u!(_mm256_slli_epi32::<17>, __m256i);
    u!(_mm256_srai_epi16::<1>, __m256i);
    u!(_mm256_srai_epi16::<15>, __m256i);
    u!(_mm256_srai_epi16::<16>, __m256i);
    u!(_mm256_srai_epi32::<31>, __m256i);
    u!(_mm256_srai_epi32::<40>, __m256i);
    u!(_mm256_castsi256_si128, __m256i);
    u!(_mm256_extracti128_si256::<0>, __m256i);
    u!(_mm256_extracti128_si256::<1>, __m256i);
    u!(_mm256_extractf128_si256::<1>, __m256i);
    un("_mm256_loadu_si256", |p: [u8; 40]| unsafe { _mm256_loadu_si256(p.as_ptr().add(5) as *const __m256i) }, |p: [u8; 40]| unsafe { _mm256_loadu_si256(p.as_ptr().add(5) as *const __m256i) });
    un(
        "_mm256_storeu_si256",
        |a: __m256i| {
            let mut o = [0u8; 40];
            unsafe { _mm256_storeu_si256(o.as_mut_ptr().add(5) as *mut __m256i, a) };
            o
        },
        |a: __m256i| {
            let mut o = [0u8; 40];
            unsafe { _mm256_storeu_si256(o.as_mut_ptr().add(5) as *mut __m256i, a) };
            o
        },
    );
    // a chain in a loop: adler32-style sums (vector values across iterations)
    bin(
        "chain_adler",
        |a: [u8; 64], b: [u8; 64]| chain_adler(a, b),
        |a: [u8; 64], b: [u8; 64]| {
            let mut h = 0i32;
            for i in 0..64 {
                h = h.wrapping_mul(31).wrapping_add((a[i] ^ b[i]) as i32);
            }
            [h, 0, 0, 0]
        },
    );
}

#[test]
fn simd_arch() {
    arch();
}
