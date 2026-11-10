//! NEON, CRC32 and SHA-1 operations. Lane zero is the lowest-address lane.
//! Hot 128-bit NEON operations use the compiler vector bridge.
//!
//! Signatures and lane operations follow stdarch's aarch64 and arm_shared/neon.
//! CRC32 uses the reflected IEEE polynomial, without initial/final complements.
//! SHA-1 follows Arm's four-round and message-schedule pseudocode.
//!
//! Unsafe entry points retain stdarch's pointer and immediate contracts: pointers
//! must cover the full load/store and lane/shift constants must be in range.
//! These functions emulate result bits, not architectural status flags.
#![allow(non_camel_case_types)]
use crate::intrinsics_vx as vx;

#[derive(Copy, Clone)]
#[repr(C, align(8))]
pub struct int8x8_t(pub [i8; 8]);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct int16x8_t(pub [i16; 8]);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct int32x4_t(pub [i32; 4]);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct int64x2_t(pub [i64; 2]);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct uint16x8_t(pub [u16; 8]);

#[derive(Copy, Clone)]
#[repr(C, align(8))]
pub struct uint16x4_t(pub [u16; 4]);

#[derive(Copy, Clone)]
#[repr(C, align(8))]
pub struct uint32x2_t(pub [u32; 2]);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct uint32x4_t(pub [u32; 4]);

#[derive(Copy, Clone)]
#[repr(C, align(8))]
pub struct uint64x1_t(pub [u64; 1]);

#[derive(Copy, Clone)]
#[repr(C, align(8))]
pub struct uint8x8_t(pub [u8; 8]);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct uint8x16_t(pub [u8; 16]);

#[derive(Copy, Clone)]
#[repr(C, align(8))]
pub struct int16x4_t(pub [i16; 4]);

#[derive(Copy, Clone)]
#[repr(C, align(8))]
pub struct int32x2_t(pub [i32; 2]);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct uint64x2_t(pub [u64; 2]);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int16x8x2_t(pub int16x8_t, pub int16x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int32x4x2_t(pub int32x4_t, pub int32x4_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint8x16x3_t(pub uint8x16_t, pub uint8x16_t, pub uint8x16_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint8x16x4_t(
    pub uint8x16_t,
    pub uint8x16_t,
    pub uint8x16_t,
    pub uint8x16_t,
);

/// This thread's FPCR (`mrs x, fpcr`): rounding mode, flush-to-zero (FZ, bit 24), default NaN
/// (DN, bit 25). A Relax builtin (no inline assembly under Relax): code shared with rustc reads
/// it under `#[cfg(relax)]` and with `asm!("mrs {0}, fpcr", ..)` otherwise.
pub unsafe fn get_fpcr() -> u64 {
    crate::intrinsics_rt::fp_env_get()
}

/// Sets this thread's FPCR (`msr fpcr, x`).
pub unsafe fn set_fpcr(x: u64) {
    crate::intrinsics_rt::fp_env_set(x)
}

#[inline(always)]
pub unsafe fn __crc32b(crc: u32, data: u8) -> u32 {
    crc32::<1>(crc, data as u64)
}

#[inline(always)]
pub unsafe fn __crc32d(crc: u32, data: u64) -> u32 {
    crc32::<8>(crc, data)
}

/// CRC-32 (reflected 0xEDB88320, no inversions) of data's low BYTES bytes into crc: the
/// instruction on aarch64 targets, bit by bit elsewhere (this module compiles on every target).
#[inline(always)]
unsafe fn crc32<const BYTES: u32>(crc: u32, data: u64) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        vx::vx_crc32(crc, data, BYTES)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut r = crc;
        let mut i = 0;
        while i < BYTES * 8 {
            r ^= ((data >> i) & 1) as u32;
            r = (r >> 1) ^ (0xedb8_8320u32 & (0u32.wrapping_sub(r & 1)));
            i += 1;
        }
        r
    }
}

/// UADDLV: the 16 lanes summed into a u16 (at most 16 * 255: it never wraps).
pub unsafe fn vaddlvq_u8(a: uint8x16_t) -> u16 {
    let mut r = 0u16;
    let mut i = 0;
    while i < 16 {
        r += a.0[i] as u16;
        i += 1;
    }
    r
}

#[inline(always)]
pub unsafe fn vaddq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vx::vx_binary(a, b, 0, 16)
}

#[inline(always)]
pub unsafe fn vaddq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vx::vx_binary(a, b, 0, 32)
}

#[inline(always)]
pub unsafe fn vaddq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 0, 32)
}

pub unsafe fn vaddvq_u32(a: uint32x4_t) -> u32 {
    let mut r = 0u32;
    let mut i = 0;
    while i < 4 {
        r = r.wrapping_add(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vaddw_u8(a: uint16x8_t, b: uint8x8_t) -> uint16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = a.0[i].wrapping_add(b.0[i] as u16);
        i += 1;
    }
    uint16x8_t(r)
}

pub unsafe fn vceq_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = if a.0[i] == b.0[i] { 255 } else { 0 };
        i += 1;
    }
    uint8x8_t(r)
}

/// TBL with one 8-byte table: an index >= 8 gives 0
pub unsafe fn vtbl1_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = if b.0[i] < 8 { a.0[b.0[i] as usize] } else { 0 };
        i += 1;
    }
    uint8x8_t(r)
}

pub unsafe fn vcgez_s8(a: int8x8_t) -> uint8x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = if a.0[i] >= 0 { 255 } else { 0 };
        i += 1;
    }
    uint8x8_t(r)
}

pub unsafe fn vcltz_s8(a: int8x8_t) -> uint8x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = if a.0[i] < 0 { 255 } else { 0 };
        i += 1;
    }
    uint8x8_t(r)
}

pub unsafe fn vcombine_s16(a: int16x4_t, b: int16x4_t) -> int16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = if i < 4 { a.0[i] } else { b.0[i - 4] };
        i += 1;
    }
    int16x8_t(r)
}

pub unsafe fn vcombine_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = if i < 4 { a.0[i] } else { b.0[i - 4] };
        i += 1;
    }
    uint16x8_t(r)
}

pub unsafe fn vcombine_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x16_t {
    let mut r = [0; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = if i < 8 { a.0[i] } else { b.0[i - 8] };
        i += 1;
    }
    uint8x16_t(r)
}

pub unsafe fn vcreate_s16(a: u64) -> int16x4_t {
    int16x4_t([
        a as i16,
        (a >> 16) as i16,
        (a >> 32) as i16,
        (a >> 48) as i16,
    ])
}

pub unsafe fn vdup_n_u8(value: u8) -> uint8x8_t {
    uint8x8_t([value; 8])
}

#[inline(always)]
pub unsafe fn vdupq_n_s16(value: i16) -> int16x8_t {
    vx::vx_splat(value as u64, 16)
}

#[inline(always)]
pub unsafe fn vdupq_n_s32(value: i32) -> int32x4_t {
    vx::vx_splat(value as u64, 32)
}

#[inline(always)]
pub unsafe fn vdupq_n_u16(value: u16) -> uint16x8_t {
    vx::vx_splat(value as u64, 16)
}

#[inline(always)]
pub unsafe fn vdupq_n_u32(value: u32) -> uint32x4_t {
    vx::vx_splat(value as u64, 32)
}

#[inline(always)]
pub unsafe fn vdupq_n_u8(value: u8) -> uint8x16_t {
    vx::vx_splat(value as u64, 8)
}

pub unsafe fn vget_high_u16(a: uint16x8_t) -> uint16x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i + 4];
        i += 1;
    }
    uint16x4_t(r)
}

pub unsafe fn vget_high_u32(a: uint32x4_t) -> uint32x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = a.0[i + 2];
        i += 1;
    }
    uint32x2_t(r)
}

pub unsafe fn vget_high_u8(a: uint8x16_t) -> uint8x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = a.0[i + 8];
        i += 1;
    }
    uint8x8_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vget_lane_u32<const LANE: i32>(v: uint32x2_t) -> u32 {
    v.0[LANE as usize]
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vget_lane_u64<const LANE: i32>(v: uint64x1_t) -> u64 {
    v.0[LANE as usize]
}

pub unsafe fn vget_low_s16(a: int16x8_t) -> int16x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i + 0];
        i += 1;
    }
    int16x4_t(r)
}

pub unsafe fn vget_low_u16(a: uint16x8_t) -> uint16x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i + 0];
        i += 1;
    }
    uint16x4_t(r)
}

pub unsafe fn vget_low_u32(a: uint32x4_t) -> uint32x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = a.0[i + 0];
        i += 1;
    }
    uint32x2_t(r)
}

pub unsafe fn vget_low_u8(a: uint8x16_t) -> uint8x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = a.0[i + 0];
        i += 1;
    }
    uint8x8_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vgetq_lane_u32<const LANE: i32>(v: uint32x4_t) -> u32 {
    v.0[LANE as usize]
}

pub unsafe fn vld1_u16(ptr: *const u16) -> uint16x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = crate::ptr::read_unaligned(ptr.add(i));
        i += 1;
    }
    uint16x4_t(r)
}

pub unsafe fn vld1_u8(ptr: *const u8) -> uint8x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = crate::ptr::read_unaligned(ptr.add(i));
        i += 1;
    }
    uint8x8_t(r)
}

#[inline(always)]
pub unsafe fn vld1q_s16(a: *const i16) -> int16x8_t {
    vx::vx_load(a as *const u8)
}

pub unsafe fn vld1q_s16_x2(a: *const i16) -> int16x8x2_t {
    int16x8x2_t(
        crate::ptr::read_unaligned(a as *const int16x8_t),
        crate::ptr::read_unaligned(a.add(8) as *const int16x8_t),
    )
}

pub unsafe fn vld1q_s32_x2(a: *const i32) -> int32x4x2_t {
    int32x4x2_t(
        crate::ptr::read_unaligned(a as *const int32x4_t),
        crate::ptr::read_unaligned(a.add(4) as *const int32x4_t),
    )
}

#[inline(always)]
pub unsafe fn vld1q_u16(a: *const u16) -> uint16x8_t {
    vx::vx_load(a as *const u8)
}

#[inline(always)]
pub unsafe fn vld1q_u32(a: *const u32) -> uint32x4_t {
    vx::vx_load(a as *const u8)
}

#[inline(always)]
pub unsafe fn vld1q_u8(a: *const u8) -> uint8x16_t {
    vx::vx_load(a as *const u8)
}

/// LD1 of four registers: 64 consecutive bytes, the first 16 in `.0`.
#[inline(always)]
pub unsafe fn vld1q_u8_x4(a: *const u8) -> uint8x16x4_t {
    uint8x16x4_t(vld1q_u8(a), vld1q_u8(a.add(16)), vld1q_u8(a.add(32)), vld1q_u8(a.add(48)))
}

#[inline(always)]
pub unsafe fn vmaxq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vx::vx_binary(a, b, 14, 16)
}

pub unsafe fn vmaxvq_u32(a: uint32x4_t) -> u32 {
    let mut r = 0u32;
    let mut i = 0;
    while i < 4 {
        r = r.max(a.0[i]);
        i += 1;
    }
    r
}

#[inline(always)]
pub unsafe fn vminq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vx::vx_binary(a, b, 12, 16)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(3))]
pub unsafe fn vmlal_high_laneq_s16<const LANE: i32>(
    a: int32x4_t,
    b: int16x8_t,
    c: int16x8_t,
) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i].wrapping_add((b.0[i + 4] as i32) * (c.0[LANE as usize] as i32));
        i += 1;
    }
    int32x4_t(r)
}

pub unsafe fn vmlal_high_u16(a: uint32x4_t, b: uint16x8_t, c: uint16x8_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i].wrapping_add((b.0[i + 4] as u32) * (c.0[i + 4] as u32));
        i += 1;
    }
    uint32x4_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(3))]
pub unsafe fn vmlal_laneq_s16<const LANE: i32>(
    a: int32x4_t,
    b: int16x4_t,
    c: int16x8_t,
) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i].wrapping_add((b.0[i + 0] as i32) * (c.0[LANE as usize] as i32));
        i += 1;
    }
    int32x4_t(r)
}

pub unsafe fn vmlal_u16(a: uint32x4_t, b: uint16x4_t, c: uint16x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i].wrapping_add((b.0[i + 0] as u32) * (c.0[i + 0] as u32));
        i += 1;
    }
    uint32x4_t(r)
}

#[inline(always)]
pub unsafe fn vmulq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vx::vx_binary(a, b, 26, 16)
}

#[inline(always)]
pub unsafe fn vmulq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vx::vx_binary(a, b, 26, 32)
}

pub unsafe fn vorr_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = a.0[i] | b.0[i];
        i += 1;
    }
    uint8x8_t(r)
}

#[inline(always)]
pub unsafe fn vorrq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vx::vx_binary(a, b, 7, 32)
}

pub unsafe fn vpadalq_u16(a: uint32x4_t, b: uint16x8_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i].wrapping_add(b.0[i * 2] as u32 + b.0[i * 2 + 1] as u32);
        i += 1;
    }
    uint32x4_t(r)
}

pub unsafe fn vpadalq_u8(a: uint16x8_t, b: uint8x16_t) -> uint16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = a.0[i].wrapping_add(b.0[i * 2] as u16 + b.0[i * 2 + 1] as u16);
        i += 1;
    }
    uint16x8_t(r)
}

pub unsafe fn vpadd_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    uint32x2_t([a.0[0].wrapping_add(a.0[1]), b.0[0].wrapping_add(b.0[1])])
}

pub unsafe fn vpaddlq_u8(a: uint8x16_t) -> uint16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = a.0[i * 2] as u16 + a.0[i * 2 + 1] as u16;
        i += 1;
    }
    uint16x8_t(r)
}

#[inline(always)]
pub unsafe fn vqmovn_s32(a: int32x4_t) -> int16x4_t {
    vx::vx_narrow(a, 21)
}

#[inline(always)]
pub unsafe fn vqmovn_u16(a: uint16x8_t) -> uint8x8_t {
    vx::vx_narrow(a, 30)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vqshlq_n_u32<const N: i32>(a: uint32x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = ((a.0[i] as u64) << N).min(0xffff_ffff) as u32;
        i += 1;
    }
    uint32x4_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vqshrun_n_s32<const N: i32>(a: int32x4_t) -> uint16x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = (a.0[i] >> N).max(0).min(65535) as u16;
        i += 1;
    }
    uint16x4_t(r)
}

pub unsafe fn vreinterpret_s8_u8(a: uint8x8_t) -> int8x8_t {
    crate::ptr::read_unaligned((&a as *const uint8x8_t).cast())
}

pub unsafe fn vreinterpret_u64_u8(a: uint8x8_t) -> uint64x1_t {
    crate::ptr::read_unaligned((&a as *const uint8x8_t).cast())
}

pub unsafe fn vreinterpretq_s32_s64(a: int64x2_t) -> int32x4_t {
    crate::ptr::read_unaligned(&a as *const int64x2_t as *const int32x4_t)
}

pub unsafe fn vreinterpretq_s32_u32(a: uint32x4_t) -> int32x4_t {
    crate::ptr::read_unaligned(&a as *const uint32x4_t as *const int32x4_t)
}

pub unsafe fn vreinterpretq_s64_s32(a: int32x4_t) -> int64x2_t {
    crate::ptr::read_unaligned(&a as *const int32x4_t as *const int64x2_t)
}

pub unsafe fn vreinterpretq_u32_s32(a: int32x4_t) -> uint32x4_t {
    crate::ptr::read_unaligned(&a as *const int32x4_t as *const uint32x4_t)
}

pub unsafe fn vreinterpretq_u32_u8(a: uint8x16_t) -> uint32x4_t {
    crate::ptr::read_unaligned(&a as *const uint8x16_t as *const uint32x4_t)
}

pub unsafe fn vrev32q_u8(a: uint8x16_t) -> uint8x16_t {
    let mut r = [0; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = a.0[(i & !3) + 3 - (i & 3)];
        i += 1;
    }
    uint8x16_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
pub unsafe fn vsetq_lane_u32<const LANE: i32>(a: u32, b: uint32x4_t) -> uint32x4_t {
    let mut r = b;
    r.0[LANE as usize] = a;
    r
}

// Hash state is [A, B, C, D]; wk already includes the SHA round constant.
#[inline(always)]
pub unsafe fn vsha1cq_u32(hash_abcd: uint32x4_t, hash_e: u32, wk: uint32x4_t) -> uint32x4_t {
    vx::vx_ternary(hash_abcd, hash_e, wk, 1)
}

#[inline(always)]
pub unsafe fn vsha1h_u32(hash_e: u32) -> u32 {
    vx::vx_sha1h(hash_e)
}

#[inline(always)]
pub unsafe fn vsha1mq_u32(hash_abcd: uint32x4_t, hash_e: u32, wk: uint32x4_t) -> uint32x4_t {
    vx::vx_ternary(hash_abcd, hash_e, wk, 3)
}

#[inline(always)]
pub unsafe fn vsha1pq_u32(hash_abcd: uint32x4_t, hash_e: u32, wk: uint32x4_t) -> uint32x4_t {
    vx::vx_ternary(hash_abcd, hash_e, wk, 2)
}

#[inline(always)]
pub unsafe fn vsha1su0q_u32(w0_3: uint32x4_t, w4_7: uint32x4_t, w8_11: uint32x4_t) -> uint32x4_t {
    vx::vx_ternary(w0_3, w4_7, w8_11, 4)
}

#[inline(always)]
pub unsafe fn vsha1su1q_u32(tw0_3: uint32x4_t, w12_15: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(tw0_3, w12_15, 80, 32)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshlq_n_u32<const N: i32>(a: uint32x4_t) -> uint32x4_t {
    vx::vx_unary(a, 1, 32, N)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshrq_n_s16<const N: i32>(a: int16x8_t) -> int16x8_t {
    vx::vx_unary(a, 3, 16, N)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshrq_n_s32<const N: i32>(a: int32x4_t) -> int32x4_t {
    vx::vx_unary(a, 3, 32, N)
}

pub unsafe fn vst1_u8(ptr: *mut u8, a: uint8x8_t) {
    let mut i = 0;
    while i < 8 {
        crate::ptr::write_unaligned(ptr.add(i), a.0[i]);
        i += 1;
    }
}

#[inline(always)]
pub unsafe fn vst1q_s16(a: *mut i16, b: int16x8_t) {
    vx::vx_store(a as *mut u8, b)
}

pub unsafe fn vst1q_s16_x2(a: *mut i16, b: int16x8x2_t) {
    vst1q_s16(a, b.0);
    vst1q_s16(a.add(8), b.1);
}

#[inline(always)]
pub unsafe fn vst1q_u32(a: *mut u32, b: uint32x4_t) {
    vx::vx_store(a as *mut u8, b)
}

pub unsafe fn vst3q_u8(a: *mut u8, b: uint8x16x3_t) {
    let mut i = 0;
    while i < 16 {
        crate::ptr::write_unaligned(a.add(i * 3 + 0), b.0 .0[i]);
        crate::ptr::write_unaligned(a.add(i * 3 + 1), b.1 .0[i]);
        crate::ptr::write_unaligned(a.add(i * 3 + 2), b.2 .0[i]);
        i += 1;
    }
}

pub unsafe fn vst4q_u8(a: *mut u8, b: uint8x16x4_t) {
    let mut i = 0;
    while i < 16 {
        crate::ptr::write_unaligned(a.add(i * 4 + 0), b.0 .0[i]);
        crate::ptr::write_unaligned(a.add(i * 4 + 1), b.1 .0[i]);
        crate::ptr::write_unaligned(a.add(i * 4 + 2), b.2 .0[i]);
        crate::ptr::write_unaligned(a.add(i * 4 + 3), b.3 .0[i]);
        i += 1;
    }
}

#[inline(always)]
pub unsafe fn vsubq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vx::vx_binary(a, b, 1, 16)
}

#[inline(always)]
pub unsafe fn vsubq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vx::vx_binary(a, b, 1, 32)
}

pub unsafe fn vtrn1q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    int64x2_t([a.0[0], b.0[0]])
}

pub unsafe fn vtrn2q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    int64x2_t([a.0[1], b.0[1]])
}

pub unsafe fn vtrnq_s32(a: int32x4_t, b: int32x4_t) -> int32x4x2_t {
    int32x4x2_t(
        int32x4_t([a.0[0], b.0[0], a.0[2], b.0[2]]),
        int32x4_t([a.0[1], b.0[1], a.0[3], b.0[3]]),
    )
}

#[inline(always)]
pub unsafe fn vzip1q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vx::vx_binary(a, b, 17, 16)
}

#[inline(always)]
pub unsafe fn vzip2q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vx::vx_binary(a, b, 18, 16)
}

// ---- SHA-256 (FEAT_SHA256), as the Arm ARM pseudocode defines SHA256H / SHA256H2 / SHA256SU0 /
// SHA256SU1 (shared/functions/crypto SHA256hash, SHAchoose, SHAmajority, SHAhashSIGMA0/1).

/// SHA256H: hash_abcd updated by four rounds.
#[inline(always)]
pub unsafe fn vsha256hq_u32(hash_abcd: uint32x4_t, hash_efgh: uint32x4_t, wk: uint32x4_t) -> uint32x4_t {
    vx::vx_ternary(hash_abcd, hash_efgh, wk, 5)
}

/// SHA256H2: hash_efgh updated by the same four rounds (hash_abcd = the pre-round abcd).
#[inline(always)]
pub unsafe fn vsha256h2q_u32(hash_efgh: uint32x4_t, hash_abcd: uint32x4_t, wk: uint32x4_t) -> uint32x4_t {
    vx::vx_ternary(hash_efgh, hash_abcd, wk, 6)
}

/// SHA256SU0: message schedule, first part.
#[inline(always)]
pub unsafe fn vsha256su0q_u32(w0_3: uint32x4_t, w4_7: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(w0_3, w4_7, 81, 32)
}

/// SHA256SU1: message schedule, second part.
#[inline(always)]
pub unsafe fn vsha256su1q_u32(tw0_3: uint32x4_t, w8_11: uint32x4_t, w12_15: uint32x4_t) -> uint32x4_t {
    vx::vx_ternary(tw0_3, w8_11, w12_15, 7)
}

#[derive(Copy, Clone)]
#[repr(C, align(8))]
pub struct float32x2_t(pub [f32; 2]);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct float32x2x2_t(pub float32x2_t, pub float32x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct float32x2x3_t(pub float32x2_t, pub float32x2_t, pub float32x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct float32x2x4_t(
    pub float32x2_t,
    pub float32x2_t,
    pub float32x2_t,
    pub float32x2_t,
);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct float32x4_t(pub [f32; 4]);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct float32x4x2_t(pub float32x4_t, pub float32x4_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct float32x4x3_t(pub float32x4_t, pub float32x4_t, pub float32x4_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct float32x4x4_t(
    pub float32x4_t,
    pub float32x4_t,
    pub float32x4_t,
    pub float32x4_t,
);

#[derive(Copy, Clone)]
#[repr(C, align(8))]
pub struct float64x1_t(pub [f64; 1]);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct float64x1x2_t(pub float64x1_t, pub float64x1_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct float64x1x3_t(pub float64x1_t, pub float64x1_t, pub float64x1_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct float64x1x4_t(
    pub float64x1_t,
    pub float64x1_t,
    pub float64x1_t,
    pub float64x1_t,
);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct float64x2_t(pub [f64; 2]);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct float64x2x2_t(pub float64x2_t, pub float64x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct float64x2x3_t(pub float64x2_t, pub float64x2_t, pub float64x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct float64x2x4_t(
    pub float64x2_t,
    pub float64x2_t,
    pub float64x2_t,
    pub float64x2_t,
);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int16x4x2_t(pub int16x4_t, pub int16x4_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int16x4x3_t(pub int16x4_t, pub int16x4_t, pub int16x4_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int16x4x4_t(pub int16x4_t, pub int16x4_t, pub int16x4_t, pub int16x4_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int16x8x3_t(pub int16x8_t, pub int16x8_t, pub int16x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int16x8x4_t(pub int16x8_t, pub int16x8_t, pub int16x8_t, pub int16x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int32x2x2_t(pub int32x2_t, pub int32x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int32x2x3_t(pub int32x2_t, pub int32x2_t, pub int32x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int32x2x4_t(pub int32x2_t, pub int32x2_t, pub int32x2_t, pub int32x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int32x4x3_t(pub int32x4_t, pub int32x4_t, pub int32x4_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int32x4x4_t(pub int32x4_t, pub int32x4_t, pub int32x4_t, pub int32x4_t);

#[derive(Copy, Clone)]
#[repr(C, align(8))]
pub struct int64x1_t(pub [i64; 1]);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int64x1x2_t(pub int64x1_t, pub int64x1_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int64x1x3_t(pub int64x1_t, pub int64x1_t, pub int64x1_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int64x1x4_t(pub int64x1_t, pub int64x1_t, pub int64x1_t, pub int64x1_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int64x2x2_t(pub int64x2_t, pub int64x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int64x2x3_t(pub int64x2_t, pub int64x2_t, pub int64x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int64x2x4_t(pub int64x2_t, pub int64x2_t, pub int64x2_t, pub int64x2_t);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct int8x16_t(pub [i8; 16]);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int8x16x2_t(pub int8x16_t, pub int8x16_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int8x16x3_t(pub int8x16_t, pub int8x16_t, pub int8x16_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int8x16x4_t(pub int8x16_t, pub int8x16_t, pub int8x16_t, pub int8x16_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int8x8x2_t(pub int8x8_t, pub int8x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int8x8x3_t(pub int8x8_t, pub int8x8_t, pub int8x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct int8x8x4_t(pub int8x8_t, pub int8x8_t, pub int8x8_t, pub int8x8_t);

#[derive(Copy, Clone)]
#[repr(C, align(8))]
pub struct poly16x4_t(pub [u16; 4]);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly16x4x2_t(pub poly16x4_t, pub poly16x4_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly16x4x3_t(pub poly16x4_t, pub poly16x4_t, pub poly16x4_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly16x4x4_t(
    pub poly16x4_t,
    pub poly16x4_t,
    pub poly16x4_t,
    pub poly16x4_t,
);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct poly16x8_t(pub [u16; 8]);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly16x8x2_t(pub poly16x8_t, pub poly16x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly16x8x3_t(pub poly16x8_t, pub poly16x8_t, pub poly16x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly16x8x4_t(
    pub poly16x8_t,
    pub poly16x8_t,
    pub poly16x8_t,
    pub poly16x8_t,
);

#[derive(Copy, Clone)]
#[repr(C, align(8))]
pub struct poly64x1_t(pub [u64; 1]);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly64x1x2_t(pub poly64x1_t, pub poly64x1_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly64x1x3_t(pub poly64x1_t, pub poly64x1_t, pub poly64x1_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly64x1x4_t(
    pub poly64x1_t,
    pub poly64x1_t,
    pub poly64x1_t,
    pub poly64x1_t,
);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct poly64x2_t(pub [u64; 2]);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly64x2x2_t(pub poly64x2_t, pub poly64x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly64x2x3_t(pub poly64x2_t, pub poly64x2_t, pub poly64x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly64x2x4_t(
    pub poly64x2_t,
    pub poly64x2_t,
    pub poly64x2_t,
    pub poly64x2_t,
);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub struct poly8x16_t(pub [u8; 16]);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly8x16x2_t(pub poly8x16_t, pub poly8x16_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly8x16x3_t(pub poly8x16_t, pub poly8x16_t, pub poly8x16_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly8x16x4_t(
    pub poly8x16_t,
    pub poly8x16_t,
    pub poly8x16_t,
    pub poly8x16_t,
);

#[derive(Copy, Clone)]
#[repr(C, align(8))]
pub struct poly8x8_t(pub [u8; 8]);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly8x8x2_t(pub poly8x8_t, pub poly8x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly8x8x3_t(pub poly8x8_t, pub poly8x8_t, pub poly8x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct poly8x8x4_t(pub poly8x8_t, pub poly8x8_t, pub poly8x8_t, pub poly8x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint16x4x2_t(pub uint16x4_t, pub uint16x4_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint16x4x3_t(pub uint16x4_t, pub uint16x4_t, pub uint16x4_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint16x4x4_t(
    pub uint16x4_t,
    pub uint16x4_t,
    pub uint16x4_t,
    pub uint16x4_t,
);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint16x8x2_t(pub uint16x8_t, pub uint16x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint16x8x3_t(pub uint16x8_t, pub uint16x8_t, pub uint16x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint16x8x4_t(
    pub uint16x8_t,
    pub uint16x8_t,
    pub uint16x8_t,
    pub uint16x8_t,
);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint32x2x2_t(pub uint32x2_t, pub uint32x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint32x2x3_t(pub uint32x2_t, pub uint32x2_t, pub uint32x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint32x2x4_t(
    pub uint32x2_t,
    pub uint32x2_t,
    pub uint32x2_t,
    pub uint32x2_t,
);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint32x4x2_t(pub uint32x4_t, pub uint32x4_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint32x4x3_t(pub uint32x4_t, pub uint32x4_t, pub uint32x4_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint32x4x4_t(
    pub uint32x4_t,
    pub uint32x4_t,
    pub uint32x4_t,
    pub uint32x4_t,
);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint64x1x2_t(pub uint64x1_t, pub uint64x1_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint64x1x3_t(pub uint64x1_t, pub uint64x1_t, pub uint64x1_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint64x1x4_t(
    pub uint64x1_t,
    pub uint64x1_t,
    pub uint64x1_t,
    pub uint64x1_t,
);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint64x2x2_t(pub uint64x2_t, pub uint64x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint64x2x3_t(pub uint64x2_t, pub uint64x2_t, pub uint64x2_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint64x2x4_t(
    pub uint64x2_t,
    pub uint64x2_t,
    pub uint64x2_t,
    pub uint64x2_t,
);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint8x16x2_t(pub uint8x16_t, pub uint8x16_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint8x8x2_t(pub uint8x8_t, pub uint8x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint8x8x3_t(pub uint8x8_t, pub uint8x8_t, pub uint8x8_t);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct uint8x8x4_t(pub uint8x8_t, pub uint8x8_t, pub uint8x8_t, pub uint8x8_t);

// Arm selects signaling NaNs before quiet NaNs, in operand order.
fn neon_snan_f32(a: f32) -> bool {
    a.is_nan() && a.to_bits() & 0x0040_0000 == 0
}

fn neon_quiet_f32(a: f32) -> f32 {
    f32::from_bits(a.to_bits() | 0x0040_0000)
}

fn neon_result_f32(a: f32, b: f32, result: f32) -> f32 {
    if neon_snan_f32(a) {
        neon_quiet_f32(a)
    } else if neon_snan_f32(b) {
        neon_quiet_f32(b)
    } else if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else if result.is_nan() {
        f32::from_bits(0x7fc0_0000)
    } else {
        result
    }
}

fn neon_add_f32(a: f32, b: f32) -> f32 {
    neon_result_f32(a, b, a + b)
}

fn neon_mul_f32(a: f32, b: f32) -> f32 {
    neon_result_f32(a, b, a * b)
}

fn neon_fma_f32(a: f32, b: f32, c: f32) -> f32 {
    let result = b.mul_add(c, a);
    if neon_snan_f32(a) {
        neon_quiet_f32(a)
    } else if neon_snan_f32(b) {
        neon_quiet_f32(b)
    } else if neon_snan_f32(c) {
        neon_quiet_f32(c)
    } else if (b.is_infinite() && c == 0.0) || (c.is_infinite() && b == 0.0) {
        f32::from_bits(0x7fc0_0000)
    } else if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else if c.is_nan() {
        c
    } else if result.is_nan() {
        f32::from_bits(0x7fc0_0000)
    } else {
        result
    }
}

fn neon_minmax_f32(a: f32, b: f32, maximum: bool, number: bool) -> f32 {
    if neon_snan_f32(a) {
        return neon_quiet_f32(a);
    }
    if neon_snan_f32(b) {
        return neon_quiet_f32(b);
    }
    if number {
        if a.is_nan() && !b.is_nan() {
            return b;
        }
        if b.is_nan() && !a.is_nan() {
            return a;
        }
    }
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    if a == b {
        return f32::from_bits(if maximum {
            a.to_bits() & b.to_bits()
        } else {
            a.to_bits() | b.to_bits()
        });
    }
    if (a > b) == maximum {
        a
    } else {
        b
    }
}

// Arm selects signaling NaNs before quiet NaNs, in operand order.
fn neon_snan_f64(a: f64) -> bool {
    a.is_nan() && a.to_bits() & 0x0008_0000_0000_0000 == 0
}

fn neon_quiet_f64(a: f64) -> f64 {
    f64::from_bits(a.to_bits() | 0x0008_0000_0000_0000)
}

fn neon_result_f64(a: f64, b: f64, result: f64) -> f64 {
    if neon_snan_f64(a) {
        neon_quiet_f64(a)
    } else if neon_snan_f64(b) {
        neon_quiet_f64(b)
    } else if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else if result.is_nan() {
        f64::from_bits(0x7ff8_0000_0000_0000)
    } else {
        result
    }
}

fn neon_fma_f64(a: f64, b: f64, c: f64) -> f64 {
    let result = b.mul_add(c, a);
    if neon_snan_f64(a) {
        neon_quiet_f64(a)
    } else if neon_snan_f64(b) {
        neon_quiet_f64(b)
    } else if neon_snan_f64(c) {
        neon_quiet_f64(c)
    } else if (b.is_infinite() && c == 0.0) || (c.is_infinite() && b == 0.0) {
        f64::from_bits(0x7ff8_0000_0000_0000)
    } else if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else if c.is_nan() {
        c
    } else if result.is_nan() {
        f64::from_bits(0x7ff8_0000_0000_0000)
    } else {
        result
    }
}

fn neon_minmax_f64(a: f64, b: f64, maximum: bool, number: bool) -> f64 {
    if neon_snan_f64(a) {
        return neon_quiet_f64(a);
    }
    if neon_snan_f64(b) {
        return neon_quiet_f64(b);
    }
    if number {
        if a.is_nan() && !b.is_nan() {
            return b;
        }
        if b.is_nan() && !a.is_nan() {
            return a;
        }
    }
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    if a == b {
        return f64::from_bits(if maximum {
            a.to_bits() & b.to_bits()
        } else {
            a.to_bits() | b.to_bits()
        });
    }
    if (a > b) == maximum {
        a
    } else {
        b
    }
}

pub unsafe fn vabsq_f32(a: float32x4_t) -> float32x4_t {
    vx::vx_unary(a, 8, 32, 0)
}

pub unsafe fn vabsq_f64(a: float64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = f64::from_bits(a.0[i].to_bits() & 0x7fffffffffffffff);
        i += 1;
    }
    float64x2_t(r)
}

#[inline(always)]
pub unsafe fn vabsq_s16(a: int16x8_t) -> int16x8_t {
    vx::vx_unary(a, 0, 16, 0)
}

#[inline(always)]
pub unsafe fn vabsq_s32(a: int32x4_t) -> int32x4_t {
    vx::vx_unary(a, 0, 32, 0)
}

#[inline(always)]
pub unsafe fn vabsq_s64(a: int64x2_t) -> int64x2_t {
    vx::vx_unary(a, 0, 64, 0)
}

#[inline(always)]
pub unsafe fn vabsq_s8(a: int8x16_t) -> int8x16_t {
    vx::vx_unary(a, 0, 8, 0)
}

#[inline(always)]
pub unsafe fn vaddq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vx::vx_binary(a, b, 60, 32)
}

pub unsafe fn vaddq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_result_f64(a.0[i], b.0[i], a.0[i] + b.0[i]);
        i += 1;
    }
    float64x2_t(r)
}

#[inline(always)]
pub unsafe fn vaddq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    vx::vx_binary(a, b, 0, 64)
}

#[inline(always)]
pub unsafe fn vaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vx::vx_binary(a, b, 0, 8)
}

#[inline(always)]
pub unsafe fn vaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 0, 16)
}

#[inline(always)]
pub unsafe fn vaddq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    vx::vx_binary(a, b, 0, 64)
}

#[inline(always)]
pub unsafe fn vaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 0, 8)
}

pub unsafe fn vaddvq_f32(a: float32x4_t) -> f32 {
    neon_add_f32(neon_add_f32(a.0[0], a.0[1]), neon_add_f32(a.0[2], a.0[3]))
}

pub unsafe fn vaddvq_s16(a: int16x8_t) -> i16 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 8 {
        r = r.wrapping_add(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vaddvq_s32(a: int32x4_t) -> i32 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 4 {
        r = r.wrapping_add(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vaddvq_u16(a: uint16x8_t) -> u16 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 8 {
        r = r.wrapping_add(a.0[i]);
        i += 1;
    }
    r
}

#[inline(always)]
pub unsafe fn vandq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vx::vx_binary(a, b, 6, 16)
}

#[inline(always)]
pub unsafe fn vandq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vx::vx_binary(a, b, 6, 32)
}

#[inline(always)]
pub unsafe fn vandq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    vx::vx_binary(a, b, 6, 64)
}

#[inline(always)]
pub unsafe fn vandq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vx::vx_binary(a, b, 6, 8)
}

#[inline(always)]
pub unsafe fn vandq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 6, 16)
}

#[inline(always)]
pub unsafe fn vandq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 6, 32)
}

#[inline(always)]
pub unsafe fn vandq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    vx::vx_binary(a, b, 6, 64)
}

#[inline(always)]
pub unsafe fn vandq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 6, 8)
}

#[inline(always)]
pub unsafe fn vbicq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 9, 32)
}

#[inline(always)]
pub unsafe fn vbslq_f32(a: uint32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    vx::vx_ternary(crate::ptr::read_unaligned(&a as *const uint32x4_t as *const float32x4_t), b, c, 10)
}

#[inline(always)]
pub unsafe fn vbslq_s16(a: uint16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t {
    vx::vx_ternary(crate::ptr::read_unaligned(&a as *const uint16x8_t as *const int16x8_t), b, c, 10)
}

#[inline(always)]
pub unsafe fn vbslq_s32(a: uint32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    vx::vx_ternary(crate::ptr::read_unaligned(&a as *const uint32x4_t as *const int32x4_t), b, c, 10)
}

#[inline(always)]
pub unsafe fn vbslq_s64(a: uint64x2_t, b: int64x2_t, c: int64x2_t) -> int64x2_t {
    vx::vx_ternary(crate::ptr::read_unaligned(&a as *const uint64x2_t as *const int64x2_t), b, c, 10)
}

#[inline(always)]
pub unsafe fn vbslq_s8(a: uint8x16_t, b: int8x16_t, c: int8x16_t) -> int8x16_t {
    vx::vx_ternary(crate::ptr::read_unaligned(&a as *const uint8x16_t as *const int8x16_t), b, c, 10)
}

#[inline(always)]
pub unsafe fn vbslq_u16(a: uint16x8_t, b: uint16x8_t, c: uint16x8_t) -> uint16x8_t {
    vx::vx_ternary(a, b, c, 10)
}

#[inline(always)]
pub unsafe fn vbslq_u32(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t) -> uint32x4_t {
    vx::vx_ternary(a, b, c, 10)
}

#[inline(always)]
pub unsafe fn vbslq_u64(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    vx::vx_ternary(a, b, c, 10)
}

#[inline(always)]
pub unsafe fn vbslq_u8(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
    vx::vx_ternary(a, b, c, 10)
}

#[inline(always)]
pub unsafe fn vceqq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 46, 32)
}

pub unsafe fn vceqq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = if a.0[i] == b.0[i] { u64::MAX } else { 0 };
        i += 1;
    }
    uint64x2_t(r)
}

#[inline(always)]
pub unsafe fn vceqq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 10, 16)
}

#[inline(always)]
pub unsafe fn vceqq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 10, 32)
}

#[inline(always)]
pub unsafe fn vceqq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    vx::vx_binary(a, b, 10, 64)
}

#[inline(always)]
pub unsafe fn vceqq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 10, 8)
}

#[inline(always)]
pub unsafe fn vceqq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 10, 16)
}

#[inline(always)]
pub unsafe fn vceqq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 10, 32)
}

#[inline(always)]
pub unsafe fn vceqq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    vx::vx_binary(a, b, 10, 64)
}

#[inline(always)]
pub unsafe fn vceqq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 10, 8)
}

pub unsafe fn vcgeq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if a.0[i] >= b.0[i] { u32::MAX } else { 0 };
        i += 1;
    }
    uint32x4_t(r)
}

pub unsafe fn vcgeq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = if a.0[i] >= b.0[i] { u64::MAX } else { 0 };
        i += 1;
    }
    uint64x2_t(r)
}

#[inline(always)]
pub unsafe fn vcgtq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 50, 32)
}

pub unsafe fn vcgtq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = if a.0[i] > b.0[i] { u64::MAX } else { 0 };
        i += 1;
    }
    uint64x2_t(r)
}

#[inline(always)]
pub unsafe fn vcgtq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 11, 16)
}

#[inline(always)]
pub unsafe fn vcgtq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 11, 32)
}

#[inline(always)]
pub unsafe fn vcgtq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    vx::vx_binary(a, b, 11, 64)
}

#[inline(always)]
pub unsafe fn vcgtq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 11, 8)
}

#[inline(always)]
pub unsafe fn vcgtq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 29, 32)
}

#[inline(always)]
pub unsafe fn vcgtq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    vx::vx_binary(a, b, 29, 64)
}

pub unsafe fn vcleq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if a.0[i] <= b.0[i] { u32::MAX } else { 0 };
        i += 1;
    }
    uint32x4_t(r)
}

pub unsafe fn vcleq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = if a.0[i] <= b.0[i] { u64::MAX } else { 0 };
        i += 1;
    }
    uint64x2_t(r)
}

#[inline(always)]
pub unsafe fn vcltq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 48, 32)
}

pub unsafe fn vcltq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = if a.0[i] < b.0[i] { u64::MAX } else { 0 };
        i += 1;
    }
    uint64x2_t(r)
}

#[inline(always)]
pub unsafe fn vcltq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    vx::vx_binary(b, a, 11, 16)
}

#[inline(always)]
pub unsafe fn vcltq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    vx::vx_binary(b, a, 11, 32)
}

#[inline(always)]
pub unsafe fn vcltq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    vx::vx_binary(b, a, 11, 64)
}

#[inline(always)]
pub unsafe fn vcltq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    vx::vx_binary(b, a, 11, 8)
}

pub unsafe fn vcombine_s8(a: int8x8_t, b: int8x8_t) -> int8x16_t {
    let mut r = [0; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = if i < 8 { a.0[i] } else { b.0[i - 8] };
        i += 1;
    }
    int8x16_t(r)
}

pub unsafe fn vcombine_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if i < 2 { a.0[i] } else { b.0[i - 2] };
        i += 1;
    }
    uint32x4_t(r)
}

pub unsafe fn vcvtnq_s32_f32(a: float32x4_t) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i].round_ties_even() as i32;
        i += 1;
    }
    int32x4_t(r)
}

pub unsafe fn vcvtq_f32_s32(a: int32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i] as f32;
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vcvtq_f64_s64(a: int64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = a.0[i] as f64;
        i += 1;
    }
    float64x2_t(r)
}

pub unsafe fn vcvtq_s32_f32(a: float32x4_t) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i] as i32;
        i += 1;
    }
    int32x4_t(r)
}

#[inline(always)]
pub unsafe fn vdivq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vx::vx_binary(a, b, 63, 32)
}

pub unsafe fn vdivq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_result_f64(a.0[i], b.0[i], a.0[i] / b.0[i]);
        i += 1;
    }
    float64x2_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vdupq_laneq_f32<const N: i32>(a: float32x4_t) -> float32x4_t {
    float32x4_t([a.0[N as usize]; 4])
}

#[inline(always)]
pub unsafe fn vdupq_n_f32(value: f32) -> float32x4_t {
    vx::vx_splat(value.to_bits() as u64, 32)
}

#[inline(always)]
pub unsafe fn vdupq_n_f64(value: f64) -> float64x2_t {
    vx::vx_splat(value.to_bits() as u64, 64)
}

#[inline(always)]
pub unsafe fn vdupq_n_s64(value: i64) -> int64x2_t {
    vx::vx_splat(value as u64, 64)
}

#[inline(always)]
pub unsafe fn vdupq_n_s8(value: i8) -> int8x16_t {
    vx::vx_splat(value as u64, 8)
}

#[inline(always)]
pub unsafe fn vdupq_n_u64(value: u64) -> uint64x2_t {
    vx::vx_splat(value as u64, 64)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vdups_laneq_f32<const N: i32>(a: float32x4_t) -> f32 {
    a.0[N as usize]
}

#[inline(always)]
pub unsafe fn veorq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vx::vx_binary(a, b, 8, 16)
}

#[inline(always)]
pub unsafe fn veorq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vx::vx_binary(a, b, 8, 32)
}

#[inline(always)]
pub unsafe fn veorq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    vx::vx_binary(a, b, 8, 64)
}

#[inline(always)]
pub unsafe fn veorq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vx::vx_binary(a, b, 8, 8)
}

#[inline(always)]
pub unsafe fn veorq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 8, 16)
}

#[inline(always)]
pub unsafe fn veorq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 8, 32)
}

#[inline(always)]
pub unsafe fn veorq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    vx::vx_binary(a, b, 8, 64)
}

#[inline(always)]
pub unsafe fn veorq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 8, 8)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
pub unsafe fn vextq_f32<const N: i32>(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if i + (N as usize) < 4 {
            a.0[i + N as usize]
        } else {
            b.0[i + N as usize - 4]
        };
        i += 1;
    }
    float32x4_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
pub unsafe fn vextq_u32<const N: i32>(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if i + (N as usize) < 4 {
            a.0[i + N as usize]
        } else {
            b.0[i + N as usize - 4]
        };
        i += 1;
    }
    uint32x4_t(r)
}

/// EXT .16b: bytes N.. of a, then the first N bytes of b (N in 0..=15): a shifted down N bytes,
/// or-ed with b shifted up 16 - N (a byte shift of 16 gives zero).
#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
#[inline(always)]
pub unsafe fn vextq_u8<const N: i32>(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vorrq_u8(vx::vx_unary(a, 5, 8, N), vx::vx_unary(b, 6, 8, 16 - N))
}

// a + b * c: stdarch's full-vector FMLA operand order; indexed FMLA uses b, c.
#[inline(always)]
pub unsafe fn vfmaq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    vx::vx_ternary(a, b, c, 8)
}

pub unsafe fn vfmaq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_fma_f64(a.0[i], c.0[i], b.0[i]);
        i += 1;
    }
    float64x2_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(3))]
pub unsafe fn vfmaq_laneq_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = neon_fma_f32(a.0[i], b.0[i], c.0[LANE as usize]);
        i += 1;
    }
    float32x4_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(3))]
pub unsafe fn vfmaq_laneq_f64<const LANE: i32>(
    a: float64x2_t,
    b: float64x2_t,
    c: float64x2_t,
) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_fma_f64(a.0[i], b.0[i], c.0[LANE as usize]);
        i += 1;
    }
    float64x2_t(r)
}

// a - b * c: stdarch's full-vector FMLS operand order.
#[inline(always)]
pub unsafe fn vfmsq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    vx::vx_ternary(a, b, c, 9)
}

pub unsafe fn vget_high_s16(a: int16x8_t) -> int16x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i + 4];
        i += 1;
    }
    int16x4_t(r)
}

pub unsafe fn vget_high_s32(a: int32x4_t) -> int32x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = a.0[i + 2];
        i += 1;
    }
    int32x2_t(r)
}

pub unsafe fn vget_high_s8(a: int8x16_t) -> int8x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = a.0[i + 8];
        i += 1;
    }
    int8x8_t(r)
}

pub unsafe fn vget_low_s32(a: int32x4_t) -> int32x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = a.0[i + 0];
        i += 1;
    }
    int32x2_t(r)
}

pub unsafe fn vget_low_s8(a: int8x16_t) -> int8x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = a.0[i + 0];
        i += 1;
    }
    int8x8_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vgetq_lane_f32<const IMM5: i32>(v: float32x4_t) -> f32 {
    v.0[IMM5 as usize]
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vgetq_lane_f64<const IMM5: i32>(a: float64x2_t) -> f64 {
    a.0[IMM5 as usize]
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vgetq_lane_s64<const IMM5: i32>(v: int64x2_t) -> i64 {
    v.0[IMM5 as usize]
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vgetq_lane_u64<const IMM5: i32>(v: uint64x2_t) -> u64 {
    v.0[IMM5 as usize]
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vgetq_lane_u8<const IMM5: i32>(v: uint8x16_t) -> u8 {
    v.0[IMM5 as usize]
}

pub unsafe fn vld1_s8(ptr: *const i8) -> int8x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = crate::ptr::read_unaligned(ptr.add(i));
        i += 1;
    }
    int8x8_t(r)
}

pub unsafe fn vld1q_dup_f32(ptr: *const f32) -> float32x4_t {
    float32x4_t([crate::ptr::read_unaligned(ptr); 4])
}

pub unsafe fn vld1q_dup_f64(ptr: *const f64) -> float64x2_t {
    float64x2_t([crate::ptr::read_unaligned(ptr); 2])
}

#[inline(always)]
pub unsafe fn vld1q_f32(ptr: *const f32) -> float32x4_t {
    vx::vx_load(ptr as *const u8)
}

#[inline(always)]
pub unsafe fn vld1q_f64(ptr: *const f64) -> float64x2_t {
    vx::vx_load(ptr as *const u8)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
pub unsafe fn vld1q_lane_f32<const LANE: i32>(ptr: *const f32, src: float32x4_t) -> float32x4_t {
    let mut r = src;
    r.0[LANE as usize] = crate::ptr::read_unaligned(ptr);
    r
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
pub unsafe fn vld1q_lane_f64<const LANE: i32>(ptr: *const f64, src: float64x2_t) -> float64x2_t {
    let mut r = src;
    r.0[LANE as usize] = crate::ptr::read_unaligned(ptr);
    r
}

#[inline(always)]
pub unsafe fn vld1q_s32(a: *const i32) -> int32x4_t {
    vx::vx_load(a as *const u8)
}

#[inline(always)]
pub unsafe fn vld1q_s8(a: *const i8) -> int8x16_t {
    vx::vx_load(a as *const u8)
}

pub unsafe fn vmaxnmq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = neon_minmax_f32(a.0[i], b.0[i], true, true);
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vmaxnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_minmax_f64(a.0[i], b.0[i], true, true);
        i += 1;
    }
    float64x2_t(r)
}

pub unsafe fn vmaxnmvq_f32(a: float32x4_t) -> f32 {
    neon_minmax_f32(
        neon_minmax_f32(a.0[0], a.0[1], true, true),
        neon_minmax_f32(a.0[2], a.0[3], true, true),
        true,
        true,
    )
}

pub unsafe fn vmaxq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vx::vx_binary(a, b, 53, 32)
}

pub unsafe fn vmaxq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_minmax_f64(a.0[i], b.0[i], true, false);
        i += 1;
    }
    float64x2_t(r)
}

#[inline(always)]
pub unsafe fn vmaxq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vx::vx_binary(a, b, 14, 8)
}

#[inline(always)]
pub unsafe fn vmaxq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 15, 16)
}

#[inline(always)]
pub unsafe fn vmaxq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 15, 32)
}

#[inline(always)]
pub unsafe fn vmaxq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 15, 8)
}

pub unsafe fn vmaxvq_s16(a: int16x8_t) -> i16 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 8 {
        r = r.max(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vmaxvq_s32(a: int32x4_t) -> i32 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 4 {
        r = r.max(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vmaxvq_s8(a: int8x16_t) -> i8 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 16 {
        r = r.max(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vmaxvq_u16(a: uint16x8_t) -> u16 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 8 {
        r = r.max(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vmaxvq_u8(a: uint8x16_t) -> u8 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 16 {
        r = r.max(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vminnmq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = neon_minmax_f32(a.0[i], b.0[i], false, true);
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vminnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_minmax_f64(a.0[i], b.0[i], false, true);
        i += 1;
    }
    float64x2_t(r)
}

pub unsafe fn vminnmvq_f32(a: float32x4_t) -> f32 {
    neon_minmax_f32(
        neon_minmax_f32(a.0[0], a.0[1], false, true),
        neon_minmax_f32(a.0[2], a.0[3], false, true),
        false,
        true,
    )
}

pub unsafe fn vminq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vx::vx_binary(a, b, 52, 32)
}

pub unsafe fn vminq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_minmax_f64(a.0[i], b.0[i], false, false);
        i += 1;
    }
    float64x2_t(r)
}

#[inline(always)]
pub unsafe fn vminq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vx::vx_binary(a, b, 12, 32)
}

#[inline(always)]
pub unsafe fn vminq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vx::vx_binary(a, b, 12, 8)
}

#[inline(always)]
pub unsafe fn vminq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 13, 16)
}

#[inline(always)]
pub unsafe fn vminq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 13, 32)
}

#[inline(always)]
pub unsafe fn vminq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 13, 8)
}

pub unsafe fn vminvq_s16(a: int16x8_t) -> i16 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 8 {
        r = r.min(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vminvq_s32(a: int32x4_t) -> i32 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 4 {
        r = r.min(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vminvq_s8(a: int8x16_t) -> i8 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 16 {
        r = r.min(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vminvq_u16(a: uint16x8_t) -> u16 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 8 {
        r = r.min(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vminvq_u32(a: uint32x4_t) -> u32 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 4 {
        r = r.min(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vminvq_u8(a: uint8x16_t) -> u8 {
    let mut r = a.0[0];
    let mut i = 1;
    while i < 16 {
        r = r.min(a.0[i]);
        i += 1;
    }
    r
}

pub unsafe fn vmlaq_s32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i].wrapping_add(b.0[i].wrapping_mul(c.0[i]));
        i += 1;
    }
    int32x4_t(r)
}

pub unsafe fn vmlsq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = neon_result_f32(
            a.0[i],
            neon_mul_f32(b.0[i], c.0[i]),
            a.0[i] - neon_mul_f32(b.0[i], c.0[i]),
        );
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vmovl_s16(a: int16x4_t) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i] as i32;
        i += 1;
    }
    int32x4_t(r)
}

pub unsafe fn vmovl_s32(a: int32x2_t) -> int64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = a.0[i] as i64;
        i += 1;
    }
    int64x2_t(r)
}

pub unsafe fn vmovl_s8(a: int8x8_t) -> int16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = a.0[i] as i16;
        i += 1;
    }
    int16x8_t(r)
}

pub unsafe fn vmovq_n_f32(value: f32) -> float32x4_t {
    float32x4_t([value; 4])
}

pub unsafe fn vmovq_n_f64(value: f64) -> float64x2_t {
    float64x2_t([value; 2])
}

pub unsafe fn vmovq_n_s16(value: i16) -> int16x8_t {
    int16x8_t([value; 8])
}

pub unsafe fn vmovq_n_s32(value: i32) -> int32x4_t {
    int32x4_t([value; 4])
}

pub unsafe fn vmovq_n_s64(value: i64) -> int64x2_t {
    int64x2_t([value; 2])
}

pub unsafe fn vmovq_n_u32(value: u32) -> uint32x4_t {
    uint32x4_t([value; 4])
}

pub unsafe fn vmull_high_s16(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = (a.0[i + 4] as i32) * (b.0[i + 4] as i32);
        i += 1;
    }
    int32x4_t(r)
}

pub unsafe fn vmull_s16(a: int16x4_t, b: int16x4_t) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = (a.0[i] as i32) * (b.0[i] as i32);
        i += 1;
    }
    int32x4_t(r)
}

pub unsafe fn vmull_s32(a: int32x2_t, b: int32x2_t) -> int64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = (a.0[i] as i64) * (b.0[i] as i64);
        i += 1;
    }
    int64x2_t(r)
}

pub unsafe fn vmull_u16(a: uint16x4_t, b: uint16x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = (a.0[i] as u32) * (b.0[i] as u32);
        i += 1;
    }
    uint32x4_t(r)
}

pub unsafe fn vmull_u32(a: uint32x2_t, b: uint32x2_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = (a.0[i] as u64) * (b.0[i] as u64);
        i += 1;
    }
    uint64x2_t(r)
}

#[inline(always)]
pub unsafe fn vmulq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vx::vx_binary(a, b, 62, 32)
}

pub unsafe fn vmulq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_result_f64(a.0[i], b.0[i], a.0[i] * b.0[i]);
        i += 1;
    }
    float64x2_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
pub unsafe fn vmulq_laneq_f32<const LANE: i32>(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = neon_result_f32(a.0[i], b.0[LANE as usize], a.0[i] * b.0[LANE as usize]);
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vmulq_n_f32(a: float32x4_t, b: f32) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = neon_result_f32(a.0[i], b, a.0[i] * b);
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vmulq_n_f64(a: float64x2_t, b: f64) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_result_f64(a.0[i], b, a.0[i] * b);
        i += 1;
    }
    float64x2_t(r)
}

#[inline(always)]
pub unsafe fn vmulq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 26, 16)
}

#[inline(always)]
pub unsafe fn vmulq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 26, 32)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
pub unsafe fn vmuls_laneq_f32<const LANE: i32>(a: f32, b: float32x4_t) -> f32 {
    neon_mul_f32(a, b.0[LANE as usize])
}

/// MVN .16b: every bit flipped
#[inline(always)]
pub unsafe fn vmvnq_u8(a: uint8x16_t) -> uint8x16_t {
    veorq_u8(a, vdupq_n_u8(0xff))
}

pub unsafe fn vmvnq_u32(a: uint32x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = !a.0[i];
        i += 1;
    }
    uint32x4_t(r)
}

#[inline(always)]
pub unsafe fn vnegq_f32(a: float32x4_t) -> float32x4_t {
    vx::vx_unary(a, 10, 32, 0)
}

pub unsafe fn vnegq_s32(a: int32x4_t) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i].wrapping_neg();
        i += 1;
    }
    int32x4_t(r)
}

#[inline(always)]
pub unsafe fn vorrq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vx::vx_binary(a, b, 7, 16)
}

#[inline(always)]
pub unsafe fn vorrq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    vx::vx_binary(a, b, 7, 64)
}

#[inline(always)]
pub unsafe fn vorrq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vx::vx_binary(a, b, 7, 8)
}

#[inline(always)]
pub unsafe fn vorrq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 7, 16)
}

#[inline(always)]
pub unsafe fn vorrq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 7, 32)
}

#[inline(always)]
pub unsafe fn vorrq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    vx::vx_binary(a, b, 7, 64)
}

#[inline(always)]
pub unsafe fn vorrq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 7, 8)
}

pub unsafe fn vpaddlq_s16(a: int16x8_t) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i * 2] as i32 + a.0[i * 2 + 1] as i32;
        i += 1;
    }
    int32x4_t(r)
}

pub unsafe fn vpaddq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if i < 2 {
            neon_add_f32(a.0[i * 2], a.0[i * 2 + 1])
        } else {
            neon_add_f32(b.0[(i - 2) * 2], b.0[(i - 2) * 2 + 1])
        };
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vpaddq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if i < 2 {
            a.0[i * 2].wrapping_add(a.0[i * 2 + 1])
        } else {
            b.0[(i - 2) * 2].wrapping_add(b.0[(i - 2) * 2 + 1])
        };
        i += 1;
    }
    int32x4_t(r)
}

pub unsafe fn vpmaxq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let mut r = [0; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = if i < 8 {
            a.0[i * 2].max(a.0[i * 2 + 1])
        } else {
            b.0[(i - 8) * 2].max(b.0[(i - 8) * 2 + 1])
        };
        i += 1;
    }
    uint8x16_t(r)
}

#[inline(always)]
pub unsafe fn vqaddq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vx::vx_binary(a, b, 2, 16)
}

#[inline(always)]
pub unsafe fn vqaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vx::vx_binary(a, b, 2, 8)
}

#[inline(always)]
pub unsafe fn vqaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 3, 16)
}

#[inline(always)]
pub unsafe fn vqaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 3, 8)
}

#[inline(always)]
pub unsafe fn vqmovn_s16(a: int16x8_t) -> int8x8_t {
    vx::vx_narrow(a, 20)
}

#[inline(always)]
pub unsafe fn vqmovun_s16(a: int16x8_t) -> uint8x8_t {
    vx::vx_narrow(a, 19)
}

pub unsafe fn vqrdmulhq_n_s16(a: int16x8_t, b: i16) -> int16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = (((a.0[i] as i64 * b as i64 * 2 + (1 << 15)) >> 16)
            .max(i16::MIN as i64)
            .min(i16::MAX as i64)) as i16;
        i += 1;
    }
    int16x8_t(r)
}

pub unsafe fn vqrdmulhq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = (((a.0[i] as i64 * b.0[i] as i64 * 2 + (1 << 15)) >> 16)
            .max(i16::MIN as i64)
            .min(i16::MAX as i64)) as i16;
        i += 1;
    }
    int16x8_t(r)
}

#[inline(always)]
pub unsafe fn vqsubq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vx::vx_binary(a, b, 4, 16)
}

#[inline(always)]
pub unsafe fn vqsubq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vx::vx_binary(a, b, 4, 8)
}

#[inline(always)]
pub unsafe fn vqsubq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 5, 16)
}

#[inline(always)]
pub unsafe fn vqsubq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 5, 8)
}

#[inline(always)]
pub unsafe fn vqtbl1q_s8(a: int8x16_t, b: uint8x16_t) -> int8x16_t {
    vx::vx_table(a, b)
}

#[inline(always)]
pub unsafe fn vqtbl1q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_table(a, b)
}

pub unsafe fn vqtbl4q_u8(a: uint8x16x4_t, b: uint8x16_t) -> uint8x16_t {
    let mut r = [0; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = {
            let index = b.0[i] as usize;
            match index / 16 {
                0 => a.0 .0[index % 16],
                1 => a.1 .0[index % 16],
                2 => a.2 .0[index % 16],
                3 => a.3 .0[index % 16],
                _ => 0,
            }
        };
        i += 1;
    }
    uint8x16_t(r)
}

pub unsafe fn vreinterpretq_f32_s32(a: int32x4_t) -> float32x4_t {
    crate::ptr::read_unaligned(&a as *const int32x4_t as *const float32x4_t)
}

pub unsafe fn vreinterpretq_f32_u32(a: uint32x4_t) -> float32x4_t {
    crate::ptr::read_unaligned(&a as *const uint32x4_t as *const float32x4_t)
}

pub unsafe fn vreinterpretq_f32_u64(a: uint64x2_t) -> float32x4_t {
    crate::ptr::read_unaligned(&a as *const uint64x2_t as *const float32x4_t)
}

pub unsafe fn vreinterpretq_f32_u8(a: uint8x16_t) -> float32x4_t {
    crate::ptr::read_unaligned(&a as *const uint8x16_t as *const float32x4_t)
}

pub unsafe fn vreinterpretq_f64_u32(a: uint32x4_t) -> float64x2_t {
    crate::ptr::read_unaligned(&a as *const uint32x4_t as *const float64x2_t)
}

pub unsafe fn vreinterpretq_f64_u64(a: uint64x2_t) -> float64x2_t {
    crate::ptr::read_unaligned(&a as *const uint64x2_t as *const float64x2_t)
}

pub unsafe fn vreinterpretq_s16_s32(a: int32x4_t) -> int16x8_t {
    crate::ptr::read_unaligned(&a as *const int32x4_t as *const int16x8_t)
}

pub unsafe fn vreinterpretq_s16_u16(a: uint16x8_t) -> int16x8_t {
    crate::ptr::read_unaligned(&a as *const uint16x8_t as *const int16x8_t)
}

pub unsafe fn vreinterpretq_s32_f32(a: float32x4_t) -> int32x4_t {
    crate::ptr::read_unaligned(&a as *const float32x4_t as *const int32x4_t)
}

pub unsafe fn vreinterpretq_s32_s16(a: int16x8_t) -> int32x4_t {
    crate::ptr::read_unaligned(&a as *const int16x8_t as *const int32x4_t)
}

pub unsafe fn vreinterpretq_s64_u64(a: uint64x2_t) -> int64x2_t {
    crate::ptr::read_unaligned(&a as *const uint64x2_t as *const int64x2_t)
}

pub unsafe fn vreinterpretq_s8_u8(a: uint8x16_t) -> int8x16_t {
    crate::ptr::read_unaligned(&a as *const uint8x16_t as *const int8x16_t)
}

pub unsafe fn vreinterpretq_u16_s16(a: int16x8_t) -> uint16x8_t {
    crate::ptr::read_unaligned(&a as *const int16x8_t as *const uint16x8_t)
}

pub unsafe fn vreinterpretq_u16_s32(a: int32x4_t) -> uint16x8_t {
    crate::ptr::read_unaligned(&a as *const int32x4_t as *const uint16x8_t)
}

pub unsafe fn vreinterpretq_u16_u32(a: uint32x4_t) -> uint16x8_t {
    crate::ptr::read_unaligned(&a as *const uint32x4_t as *const uint16x8_t)
}

pub unsafe fn vreinterpretq_u16_u8(a: uint8x16_t) -> uint16x8_t {
    crate::ptr::read_unaligned(&a as *const uint8x16_t as *const uint16x8_t)
}

pub unsafe fn vreinterpretq_u32_f32(a: float32x4_t) -> uint32x4_t {
    crate::ptr::read_unaligned(&a as *const float32x4_t as *const uint32x4_t)
}

pub unsafe fn vreinterpretq_u32_f64(a: float64x2_t) -> uint32x4_t {
    crate::ptr::read_unaligned(&a as *const float64x2_t as *const uint32x4_t)
}

pub unsafe fn vreinterpretq_u64_f32(a: float32x4_t) -> uint64x2_t {
    crate::ptr::read_unaligned(&a as *const float32x4_t as *const uint64x2_t)
}

pub unsafe fn vreinterpretq_u64_f64(a: float64x2_t) -> uint64x2_t {
    crate::ptr::read_unaligned(&a as *const float64x2_t as *const uint64x2_t)
}

pub unsafe fn vreinterpretq_u64_s64(a: int64x2_t) -> uint64x2_t {
    crate::ptr::read_unaligned(&a as *const int64x2_t as *const uint64x2_t)
}

pub unsafe fn vreinterpretq_u64_u8(a: uint8x16_t) -> uint64x2_t {
    crate::ptr::read_unaligned(&a as *const uint8x16_t as *const uint64x2_t)
}

pub unsafe fn vreinterpretq_u8_s32(a: int32x4_t) -> uint8x16_t {
    crate::ptr::read_unaligned(&a as *const int32x4_t as *const uint8x16_t)
}

pub unsafe fn vreinterpretq_u8_s8(a: int8x16_t) -> uint8x16_t {
    crate::ptr::read_unaligned(&a as *const int8x16_t as *const uint8x16_t)
}

pub unsafe fn vrev64q_f32(a: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i ^ 1];
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vrndmq_f32(a: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = neon_result_f32(a.0[i], a.0[i], a.0[i].floor());
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vrndmq_f64(a: float64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_result_f64(a.0[i], a.0[i], a.0[i].floor());
        i += 1;
    }
    float64x2_t(r)
}

pub unsafe fn vrndnq_f32(a: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = neon_result_f32(a.0[i], a.0[i], a.0[i].round_ties_even());
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vrndpq_f32(a: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = neon_result_f32(a.0[i], a.0[i], a.0[i].ceil());
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vrndpq_f64(a: float64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_result_f64(a.0[i], a.0[i], a.0[i].ceil());
        i += 1;
    }
    float64x2_t(r)
}

pub unsafe fn vrndq_f32(a: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = neon_result_f32(a.0[i], a.0[i], a.0[i].trunc());
        i += 1;
    }
    float32x4_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
pub unsafe fn vsetq_lane_f32<const LANE: i32>(a: f32, b: float32x4_t) -> float32x4_t {
    let mut r = b;
    r.0[LANE as usize] = a;
    r
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
pub unsafe fn vsetq_lane_u64<const LANE: i32>(a: u64, b: uint64x2_t) -> uint64x2_t {
    let mut r = b;
    r.0[LANE as usize] = a;
    r
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshlq_n_s32<const N: i32>(a: int32x4_t) -> int32x4_t {
    vx::vx_unary(a, 1, 32, N)
}

pub unsafe fn vshlq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = {
            let shift = b.0[i] as i8 as i32;
            if shift >= 16 {
                0
            } else if shift >= 0 {
                a.0[i] << shift
            } else if shift <= -16 {
                a.0[i] >> 15
            } else {
                a.0[i] >> -shift
            }
        };
        i += 1;
    }
    int16x8_t(r)
}

pub unsafe fn vshlq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = {
            let shift = b.0[i] as i8 as i32;
            if shift >= 32 {
                0
            } else if shift >= 0 {
                a.0[i] << shift
            } else if shift <= -32 {
                a.0[i] >> 31
            } else {
                a.0[i] >> -shift
            }
        };
        i += 1;
    }
    int32x4_t(r)
}

pub unsafe fn vshlq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = {
            let shift = b.0[i] as i8 as i32;
            if shift >= 64 {
                0
            } else if shift >= 0 {
                a.0[i] << shift
            } else if shift <= -64 {
                a.0[i] >> 63
            } else {
                a.0[i] >> -shift
            }
        };
        i += 1;
    }
    int64x2_t(r)
}

pub unsafe fn vshlq_u16(a: uint16x8_t, b: int16x8_t) -> uint16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = {
            let shift = b.0[i] as i8 as i32;
            if shift >= 16 {
                0
            } else if shift >= 0 {
                a.0[i] << shift
            } else if shift <= -16 {
                0
            } else {
                a.0[i] >> -shift
            }
        };
        i += 1;
    }
    uint16x8_t(r)
}

pub unsafe fn vshlq_u32(a: uint32x4_t, b: int32x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = {
            let shift = b.0[i] as i8 as i32;
            if shift >= 32 {
                0
            } else if shift >= 0 {
                a.0[i] << shift
            } else if shift <= -32 {
                0
            } else {
                a.0[i] >> -shift
            }
        };
        i += 1;
    }
    uint32x4_t(r)
}

pub unsafe fn vshlq_u64(a: uint64x2_t, b: int64x2_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = {
            let shift = b.0[i] as i8 as i32;
            if shift >= 64 {
                0
            } else if shift >= 0 {
                a.0[i] << shift
            } else if shift <= -64 {
                0
            } else {
                a.0[i] >> -shift
            }
        };
        i += 1;
    }
    uint64x2_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vshrn_n_u16<const N: i32>(a: uint16x8_t) -> uint8x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = (a.0[i] >> N) as u8;
        i += 1;
    }
    uint8x8_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vshrn_n_u64<const N: i32>(a: uint64x2_t) -> uint32x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = (a.0[i] >> N) as u32;
        i += 1;
    }
    uint32x2_t(r)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshrq_n_u32<const N: i32>(a: uint32x4_t) -> uint32x4_t {
    vx::vx_unary(a, 2, 32, N)
}

pub unsafe fn vsqrtq_f32(a: float32x4_t) -> float32x4_t {
    vx::vx_unary(a, 9, 32, 0)
}

pub unsafe fn vsqrtq_f64(a: float64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_result_f64(a.0[i], a.0[i], a.0[i].sqrt());
        i += 1;
    }
    float64x2_t(r)
}

#[inline(always)]
pub unsafe fn vst1q_f32(ptr: *mut f32, a: float32x4_t) {
    vx::vx_store(ptr as *mut u8, a)
}

#[inline(always)]
pub unsafe fn vst1q_f64(ptr: *mut f64, a: float64x2_t) {
    vx::vx_store(ptr as *mut u8, a)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
pub unsafe fn vst1q_lane_f32<const LANE: i32>(a: *mut f32, b: float32x4_t) {
    crate::ptr::write_unaligned(a, b.0[LANE as usize]);
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
pub unsafe fn vst1q_lane_f64<const LANE: i32>(a: *mut f64, b: float64x2_t) {
    crate::ptr::write_unaligned(a, b.0[LANE as usize]);
}

#[inline(always)]
pub unsafe fn vst1q_s32(a: *mut i32, b: int32x4_t) {
    vx::vx_store(a as *mut u8, b)
}

#[inline(always)]
pub unsafe fn vst1q_u8(a: *mut u8, b: uint8x16_t) {
    vx::vx_store(a as *mut u8, b)
}

pub unsafe fn vst3q_f32(a: *mut f32, b: float32x4x3_t) {
    let mut i = 0;
    while i < 4 {
        crate::ptr::write_unaligned(a.add(i * 3 + 0), b.0 .0[i]);
        crate::ptr::write_unaligned(a.add(i * 3 + 1), b.1 .0[i]);
        crate::ptr::write_unaligned(a.add(i * 3 + 2), b.2 .0[i]);
        i += 1;
    }
}

#[inline(always)]
pub unsafe fn vsubq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vx::vx_binary(a, b, 61, 32)
}

pub unsafe fn vsubq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    let mut r = [0.0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = neon_result_f64(a.0[i], b.0[i], a.0[i] - b.0[i]);
        i += 1;
    }
    float64x2_t(r)
}

#[inline(always)]
pub unsafe fn vsubq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    vx::vx_binary(a, b, 1, 64)
}

#[inline(always)]
pub unsafe fn vsubq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vx::vx_binary(a, b, 1, 8)
}

#[inline(always)]
pub unsafe fn vsubq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 1, 16)
}

#[inline(always)]
pub unsafe fn vsubq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 1, 32)
}

#[inline(always)]
pub unsafe fn vsubq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    vx::vx_binary(a, b, 1, 64)
}

#[inline(always)]
pub unsafe fn vsubq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 1, 8)
}

pub unsafe fn vtrn1q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if i % 2 == 0 {
            a.0[(i / 2) * 2 + 0]
        } else {
            b.0[(i / 2) * 2 + 0]
        };
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vtrn2q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if i % 2 == 0 {
            a.0[(i / 2) * 2 + 1]
        } else {
            b.0[(i / 2) * 2 + 1]
        };
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vtrnq_s16(a: int16x8_t, b: int16x8_t) -> int16x8x2_t {
    int16x8x2_t(
        {
            let mut r = [0; 8];
            let mut i = 0;
            while i < 8 {
                r[i] = if i % 2 == 0 {
                    a.0[(i / 2) * 2 + 0]
                } else {
                    b.0[(i / 2) * 2 + 0]
                };
                i += 1;
            }
            int16x8_t(r)
        },
        {
            let mut r = [0; 8];
            let mut i = 0;
            while i < 8 {
                r[i] = if i % 2 == 0 {
                    a.0[(i / 2) * 2 + 1]
                } else {
                    b.0[(i / 2) * 2 + 1]
                };
                i += 1;
            }
            int16x8_t(r)
        },
    )
}

pub unsafe fn vuzp1q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if i < 2 {
            a.0[i * 2 + 0]
        } else {
            b.0[(i - 2) * 2 + 0]
        };
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vuzp2q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if i < 2 {
            a.0[i * 2 + 1]
        } else {
            b.0[(i - 2) * 2 + 1]
        };
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vuzpq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8x2_t {
    uint16x8x2_t(
        {
            let mut r = [0; 8];
            let mut i = 0;
            while i < 8 {
                r[i] = if i < 4 {
                    a.0[i * 2 + 0]
                } else {
                    b.0[(i - 4) * 2 + 0]
                };
                i += 1;
            }
            uint16x8_t(r)
        },
        {
            let mut r = [0; 8];
            let mut i = 0;
            while i < 8 {
                r[i] = if i < 4 {
                    a.0[i * 2 + 1]
                } else {
                    b.0[(i - 4) * 2 + 1]
                };
                i += 1;
            }
            uint16x8_t(r)
        },
    )
}

pub unsafe fn vzip1q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vx::vx_binary(a, b, 17, 32)
}

pub unsafe fn vzip2q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vx::vx_binary(a, b, 18, 32)
}

#[inline(always)]
pub unsafe fn vzip2q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    vx::vx_binary(a, b, 18, 64)
}

pub unsafe fn vzip_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8x2_t {
    uint8x8x2_t(
        {
            let mut r = [0; 8];
            let mut i = 0;
            while i < 8 {
                r[i] = if i % 2 == 0 {
                    a.0[i / 2 + 0]
                } else {
                    b.0[i / 2 + 0]
                };
                i += 1;
            }
            uint8x8_t(r)
        },
        {
            let mut r = [0; 8];
            let mut i = 0;
            while i < 8 {
                r[i] = if i % 2 == 0 {
                    a.0[i / 2 + 4]
                } else {
                    b.0[i / 2 + 4]
                };
                i += 1;
            }
            uint8x8_t(r)
        },
    )
}

// a vector prints its lanes as a tuple, as stdarch's Debug impls (`uint8x16_t(3, 3, ..)`)
macro_rules! simd_debug {
    ($($t:ident)*) => {$(
        impl crate::fmt::Debug for $t {
            fn fmt(&self, f: &mut crate::fmt::Formatter<'_>) -> crate::fmt::Result {
                let mut d = f.debug_tuple(stringify!($t));
                for x in self.0.iter() {
                    d.field(x);
                }
                d.finish()
            }
        }
    )*};
}
simd_debug!(int8x8_t int16x8_t int32x4_t int64x2_t uint16x8_t uint16x4_t uint32x2_t uint32x4_t uint64x1_t uint8x8_t uint8x16_t int16x4_t int32x2_t uint64x2_t float32x2_t float32x4_t float64x1_t float64x2_t int64x1_t int8x16_t poly16x4_t poly16x8_t poly64x1_t poly64x2_t poly8x16_t poly8x8_t);

// ---------------------------------------------------------------- widening moves and integer compares (unsigned vmovl, full vcge/vcle/vcgt/vclt families)

pub unsafe fn vmovl_u8(a: uint8x8_t) -> uint16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = a.0[i] as u16;
        i += 1;
    }
    uint16x8_t(r)
}

pub unsafe fn vmovl_u16(a: uint16x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i] as u32;
        i += 1;
    }
    uint32x4_t(r)
}

pub unsafe fn vmovl_u32(a: uint32x2_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = a.0[i] as u64;
        i += 1;
    }
    uint64x2_t(r)
}

pub unsafe fn vcgeq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let mut r = [0; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = if a.0[i] >= b.0[i] { u8::MAX } else { 0 };
        i += 1;
    }
    uint8x16_t(r)
}

pub unsafe fn vcgeq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = if a.0[i] >= b.0[i] { u16::MAX } else { 0 };
        i += 1;
    }
    uint16x8_t(r)
}

pub unsafe fn vcgeq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if a.0[i] >= b.0[i] { u32::MAX } else { 0 };
        i += 1;
    }
    uint32x4_t(r)
}

pub unsafe fn vcgeq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = if a.0[i] >= b.0[i] { u64::MAX } else { 0 };
        i += 1;
    }
    uint64x2_t(r)
}

pub unsafe fn vcgeq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    let mut r = [0; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = if a.0[i] >= b.0[i] { u8::MAX } else { 0 };
        i += 1;
    }
    uint8x16_t(r)
}

pub unsafe fn vcgeq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = if a.0[i] >= b.0[i] { u16::MAX } else { 0 };
        i += 1;
    }
    uint16x8_t(r)
}

pub unsafe fn vcgeq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if a.0[i] >= b.0[i] { u32::MAX } else { 0 };
        i += 1;
    }
    uint32x4_t(r)
}

pub unsafe fn vcgeq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = if a.0[i] >= b.0[i] { u64::MAX } else { 0 };
        i += 1;
    }
    uint64x2_t(r)
}

#[inline(always)]
pub unsafe fn vcgtq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 29, 8)
}

#[inline(always)]
pub unsafe fn vcgtq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 29, 16)
}

pub unsafe fn vcleq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let mut r = [0; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = if a.0[i] <= b.0[i] { u8::MAX } else { 0 };
        i += 1;
    }
    uint8x16_t(r)
}

pub unsafe fn vcleq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = if a.0[i] <= b.0[i] { u16::MAX } else { 0 };
        i += 1;
    }
    uint16x8_t(r)
}

pub unsafe fn vcleq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if a.0[i] <= b.0[i] { u32::MAX } else { 0 };
        i += 1;
    }
    uint32x4_t(r)
}

pub unsafe fn vcleq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = if a.0[i] <= b.0[i] { u64::MAX } else { 0 };
        i += 1;
    }
    uint64x2_t(r)
}

pub unsafe fn vcleq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    let mut r = [0; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = if a.0[i] <= b.0[i] { u8::MAX } else { 0 };
        i += 1;
    }
    uint8x16_t(r)
}

pub unsafe fn vcleq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = if a.0[i] <= b.0[i] { u16::MAX } else { 0 };
        i += 1;
    }
    uint16x8_t(r)
}

pub unsafe fn vcleq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = if a.0[i] <= b.0[i] { u32::MAX } else { 0 };
        i += 1;
    }
    uint32x4_t(r)
}

pub unsafe fn vcleq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = if a.0[i] <= b.0[i] { u64::MAX } else { 0 };
        i += 1;
    }
    uint64x2_t(r)
}

#[inline(always)]
pub unsafe fn vcltq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(b, a, 29, 8)
}

#[inline(always)]
pub unsafe fn vcltq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(b, a, 29, 16)
}

#[inline(always)]
pub unsafe fn vcltq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(b, a, 29, 32)
}

#[inline(always)]
pub unsafe fn vcltq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    vx::vx_binary(b, a, 29, 64)
}

#[inline(always)]
pub unsafe fn vrhaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 16, 8)
}

#[inline(always)]
pub unsafe fn vrhaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 16, 16)
}

#[inline(always)]
pub unsafe fn vbicq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vx::vx_binary(a, b, 9, 8)
}

#[inline(always)]
pub unsafe fn vzip1q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vx::vx_binary(a, b, 17, 8)
}

#[inline(always)]
pub unsafe fn vzip2q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vx::vx_binary(a, b, 18, 8)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshlq_n_s8<const N: i32>(a: int8x16_t) -> int8x16_t {
    vx::vx_unary(a, 1, 8, N)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshrq_n_s8<const N: i32>(a: int8x16_t) -> int8x16_t {
    vx::vx_unary(a, 3, 8, N)
}

#[inline(always)]
pub unsafe fn vst1q_s8(a: *mut i8, b: int8x16_t) {
    vx::vx_store(a as *mut u8, b)
}

#[inline(always)]
pub unsafe fn vbicq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vx::vx_binary(a, b, 9, 16)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshlq_n_s16<const N: i32>(a: int16x8_t) -> int16x8_t {
    vx::vx_unary(a, 1, 16, N)
}

#[inline(always)]
pub unsafe fn vbicq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vx::vx_binary(a, b, 9, 32)
}

#[inline(always)]
pub unsafe fn vzip1q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vx::vx_binary(a, b, 17, 32)
}

#[inline(always)]
pub unsafe fn vzip2q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vx::vx_binary(a, b, 18, 32)
}

#[inline(always)]
pub unsafe fn vbicq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    vx::vx_binary(a, b, 9, 64)
}

#[inline(always)]
pub unsafe fn vzip1q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    vx::vx_binary(a, b, 17, 64)
}

#[inline(always)]
pub unsafe fn vzip2q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    vx::vx_binary(a, b, 18, 64)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshlq_n_s64<const N: i32>(a: int64x2_t) -> int64x2_t {
    vx::vx_unary(a, 1, 64, N)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshrq_n_s64<const N: i32>(a: int64x2_t) -> int64x2_t {
    vx::vx_unary(a, 3, 64, N)
}

#[inline(always)]
pub unsafe fn vld1q_s64(a: *const i64) -> int64x2_t {
    vx::vx_load(a as *const u8)
}

#[inline(always)]
pub unsafe fn vst1q_s64(a: *mut i64, b: int64x2_t) {
    vx::vx_store(a as *mut u8, b)
}

#[inline(always)]
pub unsafe fn vbicq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 9, 8)
}

#[inline(always)]
pub unsafe fn vzip1q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 17, 8)
}

#[inline(always)]
pub unsafe fn vzip2q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vx::vx_binary(a, b, 18, 8)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshlq_n_u8<const N: i32>(a: uint8x16_t) -> uint8x16_t {
    vx::vx_unary(a, 1, 8, N)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshrq_n_u8<const N: i32>(a: uint8x16_t) -> uint8x16_t {
    vx::vx_unary(a, 2, 8, N)
}

#[inline(always)]
pub unsafe fn vbicq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 9, 16)
}

#[inline(always)]
pub unsafe fn vzip1q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 17, 16)
}

#[inline(always)]
pub unsafe fn vzip2q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vx::vx_binary(a, b, 18, 16)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshlq_n_u16<const N: i32>(a: uint16x8_t) -> uint16x8_t {
    vx::vx_unary(a, 1, 16, N)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshrq_n_u16<const N: i32>(a: uint16x8_t) -> uint16x8_t {
    vx::vx_unary(a, 2, 16, N)
}

#[inline(always)]
pub unsafe fn vst1q_u16(a: *mut u16, b: uint16x8_t) {
    vx::vx_store(a as *mut u8, b)
}

#[inline(always)]
pub unsafe fn vzip1q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 17, 32)
}

#[inline(always)]
pub unsafe fn vzip2q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vx::vx_binary(a, b, 18, 32)
}

#[inline(always)]
pub unsafe fn vbicq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    vx::vx_binary(a, b, 9, 64)
}

#[inline(always)]
pub unsafe fn vzip1q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    vx::vx_binary(a, b, 17, 64)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshlq_n_u64<const N: i32>(a: uint64x2_t) -> uint64x2_t {
    vx::vx_unary(a, 1, 64, N)
}

#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
#[inline(always)]
pub unsafe fn vshrq_n_u64<const N: i32>(a: uint64x2_t) -> uint64x2_t {
    vx::vx_unary(a, 2, 64, N)
}

#[inline(always)]
pub unsafe fn vld1q_u64(a: *const u64) -> uint64x2_t {
    vx::vx_load(a as *const u8)
}

#[inline(always)]
pub unsafe fn vst1q_u64(a: *mut u64, b: uint64x2_t) {
    vx::vx_store(a as *mut u8, b)
}

#[inline(always)]
pub unsafe fn vmaxq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vx::vx_binary(a, b, 14, 32)
}

// ---- zmij 1.0 (serde_json's float formatter) and chacha20 0.10's NEON paths

/// MLA by scalar: a + b * c per lane (wrapping)
pub unsafe fn vmla_n_s32(a: int32x2_t, b: int32x2_t, c: i32) -> int32x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = a.0[i].wrapping_add(b.0[i].wrapping_mul(c));
        i += 1;
    }
    int32x2_t(r)
}

pub unsafe fn vmlaq_n_s16(a: int16x8_t, b: int16x8_t, c: i16) -> int16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = a.0[i].wrapping_add(b.0[i].wrapping_mul(c));
        i += 1;
    }
    int16x8_t(r)
}

pub unsafe fn vmlaq_n_s32(a: int32x4_t, b: int32x4_t, c: i32) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = a.0[i].wrapping_add(b.0[i].wrapping_mul(c));
        i += 1;
    }
    int32x4_t(r)
}

/// SQDMULH by scalar: the high half of 2 * a * b (a * b >> 31), saturated (only MIN * MIN
/// overflows)
pub unsafe fn vqdmulh_n_s32(a: int32x2_t, b: i32) -> int32x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = ((a.0[i] as i64 * b as i64) >> 31).min(i32::MAX as i64) as i32;
        i += 1;
    }
    int32x2_t(r)
}

pub unsafe fn vqdmulhq_n_s32(a: int32x4_t, b: i32) -> int32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = ((a.0[i] as i64 * b as i64) >> 31).min(i32::MAX as i64) as i32;
        i += 1;
    }
    int32x4_t(r)
}

pub unsafe fn vqdmulhq_n_s16(a: int16x8_t, b: i16) -> int16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = ((a.0[i] as i32 * b as i32) >> 15).min(i16::MAX as i32) as i16;
        i += 1;
    }
    int16x8_t(r)
}

/// REV64 .16b: the bytes of each 64-bit half reversed
pub unsafe fn vrev64q_u8(a: uint8x16_t) -> uint8x16_t {
    let mut r = [0; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = a.0[(i & !7) + 7 - (i & 7)];
        i += 1;
    }
    uint8x16_t(r)
}

/// REV32 .8h: the two halfwords of each 32-bit word swapped
pub unsafe fn vrev32q_u16(a: uint16x8_t) -> uint16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = a.0[i ^ 1];
        i += 1;
    }
    uint16x8_t(r)
}

/// USHLL: each u16 lane widened to u32, then shifted left N (0..=16)
#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vshll_n_u16<const N: i32>(a: uint16x4_t) -> uint32x4_t {
    let mut r = [0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = (a.0[i] as u32) << N;
        i += 1;
    }
    uint32x4_t(r)
}

/// USHR .2s by N in 1..=32 (32 gives zero)
#[cfg_attr(not(relax_check), rustc_legacy_const_generics(1))]
pub unsafe fn vshr_n_u32<const N: i32>(a: uint32x2_t) -> uint32x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = if N >= 32 { 0 } else { a.0[i] >> N };
        i += 1;
    }
    uint32x2_t(r)
}

// ---- memchr 2.8 (vrev16q_u8), simdutf8 0.1.5 (vmovq_n_u8), zmij 1.0.19+

/// REV16 .16b: the two bytes of each 16-bit lane swapped
pub unsafe fn vrev16q_u8(a: uint8x16_t) -> uint8x16_t {
    let mut r = [0; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = a.0[i ^ 1];
        i += 1;
    }
    uint8x16_t(r)
}

#[inline(always)]
pub unsafe fn vmovq_n_u8(value: u8) -> uint8x16_t {
    vdupq_n_u8(value)
}

/// CMGT #0: all ones where a lane is greater than zero
pub unsafe fn vcgtzq_s8(a: int8x16_t) -> uint8x16_t {
    let mut r = [0; 16];
    let mut i = 0;
    while i < 16 {
        r[i] = if a.0[i] > 0 { 0xff } else { 0 };
        i += 1;
    }
    uint8x16_t(r)
}

pub unsafe fn vcombine_s32(low: int32x2_t, high: int32x2_t) -> int32x4_t {
    int32x4_t([low.0[0], low.0[1], high.0[0], high.0[1]])
}

pub unsafe fn vcreate_u64(a: u64) -> uint64x1_t {
    uint64x1_t([a])
}

pub unsafe fn vdup_n_s32(value: i32) -> int32x2_t {
    int32x2_t([value; 2])
}

// ---- SHA-512 (FEAT_SHA512, part of `sha3` in rustc's feature names): sha2's
// sha512/aarch64 backends. Lane 0 is bits 63:0 of the register, lane 1 bits 127:64.

/// EXT .2d: lane N.. of a, then the first N lanes of b (N in 0..=1)
#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
pub unsafe fn vextq_u64<const N: i32>(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    if N & 1 == 0 { a } else { uint64x2_t([a.0[1], b.0[0]]) }
}

/// SHA512H Qd, Qn, Vm.2D with d = hash_ed, n = hash_gf, m = kwh_kwh2
pub unsafe fn vsha512hq_u64(hash_ed: uint64x2_t, hash_gf: uint64x2_t, kwh_kwh2: uint64x2_t) -> uint64x2_t {
    let (w, x, y) = (hash_ed.0, hash_gf.0, kwh_kwh2.0);
    let sigma1 = |v: u64| v.rotate_right(14) ^ v.rotate_right(18) ^ v.rotate_right(41);
    let hi = ((y[1] & x[0]) ^ (!y[1] & x[1])).wrapping_add(sigma1(y[1])).wrapping_add(w[1]);
    let tmp = hi.wrapping_add(y[0]);
    let lo = ((tmp & y[1]) ^ (!tmp & x[0])).wrapping_add(sigma1(tmp)).wrapping_add(w[0]);
    uint64x2_t([lo, hi])
}

/// SHA512H2 Qd, Qn, Vm.2D with d = sum_ab, n = hash_c_, m = hash_ab
pub unsafe fn vsha512h2q_u64(sum_ab: uint64x2_t, hash_c_: uint64x2_t, hash_ab: uint64x2_t) -> uint64x2_t {
    let (w, x, y) = (sum_ab.0, hash_c_.0, hash_ab.0);
    let sigma0 = |v: u64| v.rotate_right(28) ^ v.rotate_right(34) ^ v.rotate_right(39);
    let hi = ((x[0] & y[1]) ^ (x[0] & y[0]) ^ (y[1] & y[0])).wrapping_add(sigma0(y[0])).wrapping_add(w[1]);
    let lo = ((hi & y[0]) ^ (hi & y[1]) ^ (y[1] & y[0])).wrapping_add(sigma0(hi)).wrapping_add(w[0]);
    uint64x2_t([lo, hi])
}

/// SHA512SU0 Vd.2D, Vn.2D: the message schedule's sigma0 half
pub unsafe fn vsha512su0q_u64(w0_1: uint64x2_t, w2_: uint64x2_t) -> uint64x2_t {
    let (w, x) = (w0_1.0, w2_.0);
    let sig0 = |v: u64| v.rotate_right(1) ^ v.rotate_right(8) ^ (v >> 7);
    uint64x2_t([w[0].wrapping_add(sig0(w[1])), w[1].wrapping_add(sig0(x[0]))])
}

/// SHA512SU1 Vd.2D, Vn.2D, Vm.2D: the message schedule's sigma1 half
pub unsafe fn vsha512su1q_u64(s01_s02: uint64x2_t, w14_15: uint64x2_t, w9_10: uint64x2_t) -> uint64x2_t {
    let (w, x, y) = (s01_s02.0, w14_15.0, w9_10.0);
    let sig1 = |v: u64| v.rotate_right(19) ^ v.rotate_right(61) ^ (v >> 6);
    uint64x2_t([w[0].wrapping_add(sig1(x[0])).wrapping_add(y[0]), w[1].wrapping_add(sig1(x[1])).wrapping_add(y[1])])
}

// ---- AES (FEAT_AES, `aes`): the aes crate's aarch64 backend. A state is FIPS-197's in
// column-major memory order: byte i is row i % 4 of column i / 4.

/// AESE: SubBytes(ShiftRows(data ^ key)); the round's MixColumns is a separate AESMC
pub unsafe fn vaeseq_u8(data: uint8x16_t, key: uint8x16_t) -> uint8x16_t {
    let mut r = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        let row = i % 4;
        let j = ((i / 4 + row) % 4) * 4 + row;
        r[i] = super::x86_64::AES_SBOXES[0][(data.0[j] ^ key.0[j]) as usize];
        i += 1;
    }
    uint8x16_t(r)
}

/// AESD: InvSubBytes(InvShiftRows(data ^ key))
pub unsafe fn vaesdq_u8(data: uint8x16_t, key: uint8x16_t) -> uint8x16_t {
    let mut r = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        let row = i % 4;
        let j = ((i / 4 + 4 - row) % 4) * 4 + row;
        r[i] = super::x86_64::AES_SBOXES[1][(data.0[j] ^ key.0[j]) as usize];
        i += 1;
    }
    uint8x16_t(r)
}

/// times x modulo x^8 + x^4 + x^3 + x + 1
#[inline(always)]
fn aes_xtime(a: u8) -> u8 {
    (a << 1) ^ ((a >> 7) * 0x1b)
}

/// AESMC: MixColumns, each column times {03}x^3 + {01}x^2 + {01}x + {02}
pub unsafe fn vaesmcq_u8(data: uint8x16_t) -> uint8x16_t {
    let a = data.0;
    let mut r = [0u8; 16];
    let mut c = 0;
    while c < 16 {
        let t = a[c] ^ a[c + 1] ^ a[c + 2] ^ a[c + 3];
        r[c] = a[c] ^ t ^ aes_xtime(a[c] ^ a[c + 1]);
        r[c + 1] = a[c + 1] ^ t ^ aes_xtime(a[c + 1] ^ a[c + 2]);
        r[c + 2] = a[c + 2] ^ t ^ aes_xtime(a[c + 2] ^ a[c + 3]);
        r[c + 3] = a[c + 3] ^ t ^ aes_xtime(a[c + 3] ^ a[c]);
        c += 4;
    }
    uint8x16_t(r)
}

/// AESIMC: InvMixColumns, as MixColumns of the column times {04}x^2 + {05}
pub unsafe fn vaesimcq_u8(data: uint8x16_t) -> uint8x16_t {
    let mut a = data.0;
    let mut c = 0;
    while c < 16 {
        let u = aes_xtime(aes_xtime(a[c] ^ a[c + 2]));
        let v = aes_xtime(aes_xtime(a[c + 1] ^ a[c + 3]));
        a[c] ^= u;
        a[c + 1] ^= v;
        a[c + 2] ^= u;
        a[c + 3] ^= v;
        c += 4;
    }
    vaesmcq_u8(uint8x16_t(a))
}

// ---- PMULL (FEAT_PMULL, under `aes` in rustc's feature names): polyval's and ghash's
// carry-less multiply.

/// stdarch's polynomial element types (`vmull_p64`'s operands; p128 is its result)
pub type p8 = u8;
pub type p16 = u16;
pub type p64 = u64;

/// PMULL .1q: the 128-bit carry-less product of two 64-bit polynomials
pub unsafe fn vmull_p64(a: p64, b: p64) -> p128 {
    let r: uint64x2_t = vx::vx_binary(uint64x2_t([a, 0]), uint64x2_t([b, 0]), 90, 64);
    (r.0[0] as u128) | (r.0[1] as u128) << 64
}

/// LD1 of two registers: 32 consecutive bytes, the first 16 in `.0` (constant_time_eq).
#[inline(always)]
pub unsafe fn vld1q_u8_x2(a: *const u8) -> uint8x16x2_t {
    uint8x16x2_t(vld1q_u8(a), vld1q_u8(a.add(16)))
}

/// LD1 of three registers: 48 consecutive bytes.
#[inline(always)]
pub unsafe fn vld1q_u8_x3(a: *const u8) -> uint8x16x3_t {
    uint8x16x3_t(vld1q_u8(a), vld1q_u8(a.add(16)), vld1q_u8(a.add(32)))
}

// ---- xxh3's accumulate and scramble (xxhash-rust, redb): 32 x 32 -> 64 multiply-accumulate
// over the even and odd halves of the lanes.

pub unsafe fn vdup_n_u32(value: u32) -> uint32x2_t {
    uint32x2_t([value; 2])
}

/// UMLAL: a + b * c, each 32-bit lane widened to 64
pub unsafe fn vmlal_u32(a: uint64x2_t, b: uint32x2_t, c: uint32x2_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = a.0[i].wrapping_add((b.0[i] as u64) * (c.0[i] as u64));
        i += 1;
    }
    uint64x2_t(r)
}

/// UMLAL2: the same over the high halves of b and c
pub unsafe fn vmlal_high_u32(a: uint64x2_t, b: uint32x4_t, c: uint32x4_t) -> uint64x2_t {
    let mut r = [0; 2];
    let mut i = 0;
    while i < 2 {
        r[i] = a.0[i].wrapping_add((b.0[i + 2] as u64) * (c.0[i + 2] as u64));
        i += 1;
    }
    uint64x2_t(r)
}

/// XTN: the low half of each 64-bit lane
pub unsafe fn vmovn_u64(a: uint64x2_t) -> uint32x2_t {
    uint32x2_t([a.0[0] as u32, a.0[1] as u32])
}

/// UZP1 and UZP2: the even lanes of a then b in `.0`, the odd lanes in `.1`
pub unsafe fn vuzpq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4x2_t {
    uint32x4x2_t(
        uint32x4_t([a.0[0], a.0[2], b.0[0], b.0[2]]),
        uint32x4_t([a.0[1], a.0[3], b.0[1], b.0[3]]),
    )
}

// ---- zlib-rs: the CRC-32 half-word and word forms, adler32's widening add of the high half,
// compare256's deinterleaving load and shift-right-insert.

#[inline(always)]
pub unsafe fn __crc32h(crc: u32, data: u16) -> u32 {
    crc32::<2>(crc, data as u64)
}

#[inline(always)]
pub unsafe fn __crc32w(crc: u32, data: u32) -> u32 {
    crc32::<4>(crc, data as u64)
}

/// UADDW2: a plus the high eight bytes of b, widened
pub unsafe fn vaddw_high_u8(a: uint16x8_t, b: uint8x16_t) -> uint16x8_t {
    let mut r = [0; 8];
    let mut i = 0;
    while i < 8 {
        r[i] = a.0[i].wrapping_add(b.0[i + 8] as u16);
        i += 1;
    }
    uint16x8_t(r)
}

/// LD4: 64 bytes deinterleaved, byte 4 * i + k in lane i of register k
pub unsafe fn vld4q_u8(a: *const u8) -> uint8x16x4_t {
    let mut r = [[0u8; 16]; 4];
    let mut i = 0;
    while i < 64 {
        r[i % 4][i / 4] = *a.add(i);
        i += 1;
    }
    uint8x16x4_t(uint8x16_t(r[0]), uint8x16_t(r[1]), uint8x16_t(r[2]), uint8x16_t(r[3]))
}

/// SRI: b shifted right by N (1..=8) into a, whose top N bits of each lane stay
#[cfg_attr(not(relax_check), rustc_legacy_const_generics(2))]
pub unsafe fn vsriq_n_u8<const N: i32>(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let keep = !(0xffu32 >> N) as u8;
    vorrq_u8(vandq_u8(a, vdupq_n_u8(keep)), vshrq_n_u8::<N>(b))
}

// ---- Reciprocal and reciprocal square root estimates and their Newton steps (tiny-skia's
// f32x4 recip_fast / recip_sqrt): Arm's FPRecipEstimate, FPRSqrtEstimate, FPRecipStepFused and
// FPRSqrtStepFused under the default FPCR (round to nearest, no flush to zero, NaNs propagate).

/// FRECPE's lane: an 8-bit estimate of 1 / a
fn neon_recpe_f32(a: f32) -> f32 {
    let bits = a.to_bits();
    let sign = bits & 0x8000_0000;
    let mut exp = ((bits >> 23) & 0xff) as i32;
    let mut frac = bits & 0x007f_ffff;
    if exp == 0xff {
        return if frac != 0 { neon_quiet_f32(a) } else { f32::from_bits(sign) };
    }
    if exp == 0 {
        // zero, and anything below 2^-128 (its reciprocal overflows): infinity
        if frac < 0x0020_0000 {
            return f32::from_bits(sign | 0x7f80_0000);
        }
        // a subnormal of 2^-128 or more: normalize it by one or two places
        if frac & 0x0040_0000 == 0 {
            exp = -1;
            frac = (frac << 2) & 0x007f_ffff;
        } else {
            frac = (frac << 1) & 0x007f_ffff;
        }
    }
    // the operand scaled to [0.5, 1) in 1/512ths, its estimate in [1, 2) in 1/256ths
    let scaled = 256 | (frac >> 15);
    let estimate = ((1 << 19) / (scaled * 2 + 1) + 1) / 2;
    let mut result_exp = 253 - exp;
    let mut result = (estimate & 0xff) << 15;
    // 2^126 and above: the estimate is subnormal
    if result_exp == 0 {
        result = 0x0040_0000 | result >> 1;
    } else if result_exp == -1 {
        result = 0x0020_0000 | result >> 2;
        result_exp = 0;
    }
    f32::from_bits(sign | (result_exp as u32) << 23 | result)
}

/// Arm's RecipSqrtEstimate for each 9-bit operand 128..=511 (0.25 to 1 in 1/512ths): the low
/// eight bits of the estimate in 256..=511 (1 to 2 in 1/256ths)
const fn neon_rsqrt_estimates() -> [u8; 384] {
    let mut table = [0u8; 384];
    let mut i = 0;
    while i < 384 {
        let a = i as u32 + 128;
        let a = if a < 256 { a * 2 + 1 } else { ((a >> 1 << 1) + 1) * 2 };
        let mut b = 512;
        while a * (b + 1) * (b + 1) < 1 << 28 {
            b += 1;
        }
        table[i] = ((b + 1) / 2) as u8;
        i += 1;
    }
    table
}

const NEON_RSQRT_ESTIMATES: [u8; 384] = neon_rsqrt_estimates();

/// FRSQRTE's lane: an 8-bit estimate of 1 / sqrt(a)
fn neon_rsqrte_f32(a: f32) -> f32 {
    let bits = a.to_bits();
    let mut exp = ((bits >> 23) & 0xff) as i32;
    let mut frac = bits & 0x007f_ffff;
    if exp == 0xff && frac != 0 {
        return neon_quiet_f32(a);
    }
    if exp == 0 && frac == 0 {
        return f32::from_bits((bits & 0x8000_0000) | 0x7f80_0000);
    }
    if bits >> 31 != 0 {
        return f32::from_bits(0x7fc0_0000);
    }
    if exp == 0xff {
        return 0.0;
    }
    if exp == 0 {
        while frac & 0x0040_0000 == 0 {
            frac <<= 1;
            exp -= 1;
        }
        frac = (frac << 1) & 0x007f_ffff;
    }
    // the operand scaled to [0.25, 1) with its exponent's parity kept
    let scaled = if exp & 1 == 0 { 256 | (frac >> 15) } else { 128 | (frac >> 16) };
    let estimate = NEON_RSQRT_ESTIMATES[(scaled - 128) as usize] as u32;
    f32::from_bits((((380 - exp) / 2) as u32) << 23 | estimate << 15)
}

/// FRECPS's and FRSQRTS's lane: 2 - a * b, or (3 - a * b) / 2, fused. a is negated before the
/// NaN rules (a NaN taken from a has its sign flipped); infinity times zero gives 2, or 1.5.
fn neon_step_f32(a: f32, b: f32, rsqrt: bool) -> f32 {
    let a = f32::from_bits(a.to_bits() ^ 0x8000_0000);
    if a.is_nan() || b.is_nan() {
        return neon_result_f32(a, b, 0.0);
    }
    if (a.is_infinite() && b == 0.0) || (a == 0.0 && b.is_infinite()) {
        return if rsqrt { 1.5 } else { 2.0 };
    }
    if !rsqrt {
        return a.mul_add(b, 2.0);
    }
    // 1.5 + (a / 2) * b in one rounding. Halving the larger operand is exact unless it is
    // below 2^-125, where the product is far below 1.5's last place whichever way it rounds.
    let (a, b) = if a.abs() >= b.abs() { (a * 0.5, b) } else { (a, b * 0.5) };
    a.mul_add(b, 1.5)
}

pub unsafe fn vrecpeq_f32(a: float32x4_t) -> float32x4_t {
    float32x4_t([neon_recpe_f32(a.0[0]), neon_recpe_f32(a.0[1]), neon_recpe_f32(a.0[2]), neon_recpe_f32(a.0[3])])
}

pub unsafe fn vrecpsq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = neon_step_f32(a.0[i], b.0[i], false);
        i += 1;
    }
    float32x4_t(r)
}

pub unsafe fn vrsqrteq_f32(a: float32x4_t) -> float32x4_t {
    float32x4_t([neon_rsqrte_f32(a.0[0]), neon_rsqrte_f32(a.0[1]), neon_rsqrte_f32(a.0[2]), neon_rsqrte_f32(a.0[3])])
}

pub unsafe fn vrsqrtsq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut r = [0.0; 4];
    let mut i = 0;
    while i < 4 {
        r[i] = neon_step_f32(a.0[i], b.0[i], true);
        i += 1;
    }
    float32x4_t(r)
}

// ---- vreinterpret{,q}_<to>_<from>: every pair of core::arch's element types (stdarch
// generates the same set); the bits are unchanged.

/// stdarch's 128-bit polynomial (`vreinterpretq_p128_*`, `vmull_p64`'s result)
pub type p128 = u128;

#[inline(always)]
pub unsafe fn vreinterpret_s8_s16(a: int16x4_t) -> int8x8_t {
    crate::ptr::read_unaligned((&a as *const int16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s8_s32(a: int32x2_t) -> int8x8_t {
    crate::ptr::read_unaligned((&a as *const int32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s8_s64(a: int64x1_t) -> int8x8_t {
    crate::ptr::read_unaligned((&a as *const int64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s8_u16(a: uint16x4_t) -> int8x8_t {
    crate::ptr::read_unaligned((&a as *const uint16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s8_u32(a: uint32x2_t) -> int8x8_t {
    crate::ptr::read_unaligned((&a as *const uint32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s8_u64(a: uint64x1_t) -> int8x8_t {
    crate::ptr::read_unaligned((&a as *const uint64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s8_f32(a: float32x2_t) -> int8x8_t {
    crate::ptr::read_unaligned((&a as *const float32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s8_f64(a: float64x1_t) -> int8x8_t {
    crate::ptr::read_unaligned((&a as *const float64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s8_p8(a: poly8x8_t) -> int8x8_t {
    crate::ptr::read_unaligned((&a as *const poly8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s8_p16(a: poly16x4_t) -> int8x8_t {
    crate::ptr::read_unaligned((&a as *const poly16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s8_p64(a: poly64x1_t) -> int8x8_t {
    crate::ptr::read_unaligned((&a as *const poly64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s16_s8(a: int8x8_t) -> int16x4_t {
    crate::ptr::read_unaligned((&a as *const int8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s16_s32(a: int32x2_t) -> int16x4_t {
    crate::ptr::read_unaligned((&a as *const int32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s16_s64(a: int64x1_t) -> int16x4_t {
    crate::ptr::read_unaligned((&a as *const int64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s16_u8(a: uint8x8_t) -> int16x4_t {
    crate::ptr::read_unaligned((&a as *const uint8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s16_u16(a: uint16x4_t) -> int16x4_t {
    crate::ptr::read_unaligned((&a as *const uint16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s16_u32(a: uint32x2_t) -> int16x4_t {
    crate::ptr::read_unaligned((&a as *const uint32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s16_u64(a: uint64x1_t) -> int16x4_t {
    crate::ptr::read_unaligned((&a as *const uint64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s16_f32(a: float32x2_t) -> int16x4_t {
    crate::ptr::read_unaligned((&a as *const float32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s16_f64(a: float64x1_t) -> int16x4_t {
    crate::ptr::read_unaligned((&a as *const float64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s16_p8(a: poly8x8_t) -> int16x4_t {
    crate::ptr::read_unaligned((&a as *const poly8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s16_p16(a: poly16x4_t) -> int16x4_t {
    crate::ptr::read_unaligned((&a as *const poly16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s16_p64(a: poly64x1_t) -> int16x4_t {
    crate::ptr::read_unaligned((&a as *const poly64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s32_s8(a: int8x8_t) -> int32x2_t {
    crate::ptr::read_unaligned((&a as *const int8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s32_s16(a: int16x4_t) -> int32x2_t {
    crate::ptr::read_unaligned((&a as *const int16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s32_s64(a: int64x1_t) -> int32x2_t {
    crate::ptr::read_unaligned((&a as *const int64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s32_u8(a: uint8x8_t) -> int32x2_t {
    crate::ptr::read_unaligned((&a as *const uint8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s32_u16(a: uint16x4_t) -> int32x2_t {
    crate::ptr::read_unaligned((&a as *const uint16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s32_u32(a: uint32x2_t) -> int32x2_t {
    crate::ptr::read_unaligned((&a as *const uint32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s32_u64(a: uint64x1_t) -> int32x2_t {
    crate::ptr::read_unaligned((&a as *const uint64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s32_f32(a: float32x2_t) -> int32x2_t {
    crate::ptr::read_unaligned((&a as *const float32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s32_f64(a: float64x1_t) -> int32x2_t {
    crate::ptr::read_unaligned((&a as *const float64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s32_p8(a: poly8x8_t) -> int32x2_t {
    crate::ptr::read_unaligned((&a as *const poly8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s32_p16(a: poly16x4_t) -> int32x2_t {
    crate::ptr::read_unaligned((&a as *const poly16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s32_p64(a: poly64x1_t) -> int32x2_t {
    crate::ptr::read_unaligned((&a as *const poly64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s64_s8(a: int8x8_t) -> int64x1_t {
    crate::ptr::read_unaligned((&a as *const int8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s64_s16(a: int16x4_t) -> int64x1_t {
    crate::ptr::read_unaligned((&a as *const int16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s64_s32(a: int32x2_t) -> int64x1_t {
    crate::ptr::read_unaligned((&a as *const int32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s64_u8(a: uint8x8_t) -> int64x1_t {
    crate::ptr::read_unaligned((&a as *const uint8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s64_u16(a: uint16x4_t) -> int64x1_t {
    crate::ptr::read_unaligned((&a as *const uint16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s64_u32(a: uint32x2_t) -> int64x1_t {
    crate::ptr::read_unaligned((&a as *const uint32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s64_u64(a: uint64x1_t) -> int64x1_t {
    crate::ptr::read_unaligned((&a as *const uint64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s64_f32(a: float32x2_t) -> int64x1_t {
    crate::ptr::read_unaligned((&a as *const float32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s64_f64(a: float64x1_t) -> int64x1_t {
    crate::ptr::read_unaligned((&a as *const float64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s64_p8(a: poly8x8_t) -> int64x1_t {
    crate::ptr::read_unaligned((&a as *const poly8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s64_p16(a: poly16x4_t) -> int64x1_t {
    crate::ptr::read_unaligned((&a as *const poly16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_s64_p64(a: poly64x1_t) -> int64x1_t {
    crate::ptr::read_unaligned((&a as *const poly64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u8_s8(a: int8x8_t) -> uint8x8_t {
    crate::ptr::read_unaligned((&a as *const int8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u8_s16(a: int16x4_t) -> uint8x8_t {
    crate::ptr::read_unaligned((&a as *const int16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u8_s32(a: int32x2_t) -> uint8x8_t {
    crate::ptr::read_unaligned((&a as *const int32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u8_s64(a: int64x1_t) -> uint8x8_t {
    crate::ptr::read_unaligned((&a as *const int64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u8_u16(a: uint16x4_t) -> uint8x8_t {
    crate::ptr::read_unaligned((&a as *const uint16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u8_u32(a: uint32x2_t) -> uint8x8_t {
    crate::ptr::read_unaligned((&a as *const uint32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u8_u64(a: uint64x1_t) -> uint8x8_t {
    crate::ptr::read_unaligned((&a as *const uint64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u8_f32(a: float32x2_t) -> uint8x8_t {
    crate::ptr::read_unaligned((&a as *const float32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u8_f64(a: float64x1_t) -> uint8x8_t {
    crate::ptr::read_unaligned((&a as *const float64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u8_p8(a: poly8x8_t) -> uint8x8_t {
    crate::ptr::read_unaligned((&a as *const poly8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u8_p16(a: poly16x4_t) -> uint8x8_t {
    crate::ptr::read_unaligned((&a as *const poly16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u8_p64(a: poly64x1_t) -> uint8x8_t {
    crate::ptr::read_unaligned((&a as *const poly64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u16_s8(a: int8x8_t) -> uint16x4_t {
    crate::ptr::read_unaligned((&a as *const int8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u16_s16(a: int16x4_t) -> uint16x4_t {
    crate::ptr::read_unaligned((&a as *const int16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u16_s32(a: int32x2_t) -> uint16x4_t {
    crate::ptr::read_unaligned((&a as *const int32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u16_s64(a: int64x1_t) -> uint16x4_t {
    crate::ptr::read_unaligned((&a as *const int64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u16_u8(a: uint8x8_t) -> uint16x4_t {
    crate::ptr::read_unaligned((&a as *const uint8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u16_u32(a: uint32x2_t) -> uint16x4_t {
    crate::ptr::read_unaligned((&a as *const uint32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u16_u64(a: uint64x1_t) -> uint16x4_t {
    crate::ptr::read_unaligned((&a as *const uint64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u16_f32(a: float32x2_t) -> uint16x4_t {
    crate::ptr::read_unaligned((&a as *const float32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u16_f64(a: float64x1_t) -> uint16x4_t {
    crate::ptr::read_unaligned((&a as *const float64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u16_p8(a: poly8x8_t) -> uint16x4_t {
    crate::ptr::read_unaligned((&a as *const poly8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u16_p16(a: poly16x4_t) -> uint16x4_t {
    crate::ptr::read_unaligned((&a as *const poly16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u16_p64(a: poly64x1_t) -> uint16x4_t {
    crate::ptr::read_unaligned((&a as *const poly64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u32_s8(a: int8x8_t) -> uint32x2_t {
    crate::ptr::read_unaligned((&a as *const int8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u32_s16(a: int16x4_t) -> uint32x2_t {
    crate::ptr::read_unaligned((&a as *const int16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u32_s32(a: int32x2_t) -> uint32x2_t {
    crate::ptr::read_unaligned((&a as *const int32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u32_s64(a: int64x1_t) -> uint32x2_t {
    crate::ptr::read_unaligned((&a as *const int64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u32_u8(a: uint8x8_t) -> uint32x2_t {
    crate::ptr::read_unaligned((&a as *const uint8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u32_u16(a: uint16x4_t) -> uint32x2_t {
    crate::ptr::read_unaligned((&a as *const uint16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u32_u64(a: uint64x1_t) -> uint32x2_t {
    crate::ptr::read_unaligned((&a as *const uint64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u32_f32(a: float32x2_t) -> uint32x2_t {
    crate::ptr::read_unaligned((&a as *const float32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u32_f64(a: float64x1_t) -> uint32x2_t {
    crate::ptr::read_unaligned((&a as *const float64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u32_p8(a: poly8x8_t) -> uint32x2_t {
    crate::ptr::read_unaligned((&a as *const poly8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u32_p16(a: poly16x4_t) -> uint32x2_t {
    crate::ptr::read_unaligned((&a as *const poly16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u32_p64(a: poly64x1_t) -> uint32x2_t {
    crate::ptr::read_unaligned((&a as *const poly64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u64_s8(a: int8x8_t) -> uint64x1_t {
    crate::ptr::read_unaligned((&a as *const int8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u64_s16(a: int16x4_t) -> uint64x1_t {
    crate::ptr::read_unaligned((&a as *const int16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u64_s32(a: int32x2_t) -> uint64x1_t {
    crate::ptr::read_unaligned((&a as *const int32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u64_s64(a: int64x1_t) -> uint64x1_t {
    crate::ptr::read_unaligned((&a as *const int64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u64_u16(a: uint16x4_t) -> uint64x1_t {
    crate::ptr::read_unaligned((&a as *const uint16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u64_u32(a: uint32x2_t) -> uint64x1_t {
    crate::ptr::read_unaligned((&a as *const uint32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u64_f32(a: float32x2_t) -> uint64x1_t {
    crate::ptr::read_unaligned((&a as *const float32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u64_f64(a: float64x1_t) -> uint64x1_t {
    crate::ptr::read_unaligned((&a as *const float64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u64_p8(a: poly8x8_t) -> uint64x1_t {
    crate::ptr::read_unaligned((&a as *const poly8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u64_p16(a: poly16x4_t) -> uint64x1_t {
    crate::ptr::read_unaligned((&a as *const poly16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_u64_p64(a: poly64x1_t) -> uint64x1_t {
    crate::ptr::read_unaligned((&a as *const poly64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f32_s8(a: int8x8_t) -> float32x2_t {
    crate::ptr::read_unaligned((&a as *const int8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f32_s16(a: int16x4_t) -> float32x2_t {
    crate::ptr::read_unaligned((&a as *const int16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f32_s32(a: int32x2_t) -> float32x2_t {
    crate::ptr::read_unaligned((&a as *const int32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f32_s64(a: int64x1_t) -> float32x2_t {
    crate::ptr::read_unaligned((&a as *const int64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f32_u8(a: uint8x8_t) -> float32x2_t {
    crate::ptr::read_unaligned((&a as *const uint8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f32_u16(a: uint16x4_t) -> float32x2_t {
    crate::ptr::read_unaligned((&a as *const uint16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f32_u32(a: uint32x2_t) -> float32x2_t {
    crate::ptr::read_unaligned((&a as *const uint32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f32_u64(a: uint64x1_t) -> float32x2_t {
    crate::ptr::read_unaligned((&a as *const uint64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f32_f64(a: float64x1_t) -> float32x2_t {
    crate::ptr::read_unaligned((&a as *const float64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f32_p8(a: poly8x8_t) -> float32x2_t {
    crate::ptr::read_unaligned((&a as *const poly8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f32_p16(a: poly16x4_t) -> float32x2_t {
    crate::ptr::read_unaligned((&a as *const poly16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f32_p64(a: poly64x1_t) -> float32x2_t {
    crate::ptr::read_unaligned((&a as *const poly64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f64_s8(a: int8x8_t) -> float64x1_t {
    crate::ptr::read_unaligned((&a as *const int8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f64_s16(a: int16x4_t) -> float64x1_t {
    crate::ptr::read_unaligned((&a as *const int16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f64_s32(a: int32x2_t) -> float64x1_t {
    crate::ptr::read_unaligned((&a as *const int32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f64_s64(a: int64x1_t) -> float64x1_t {
    crate::ptr::read_unaligned((&a as *const int64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f64_u8(a: uint8x8_t) -> float64x1_t {
    crate::ptr::read_unaligned((&a as *const uint8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f64_u16(a: uint16x4_t) -> float64x1_t {
    crate::ptr::read_unaligned((&a as *const uint16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f64_u32(a: uint32x2_t) -> float64x1_t {
    crate::ptr::read_unaligned((&a as *const uint32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f64_u64(a: uint64x1_t) -> float64x1_t {
    crate::ptr::read_unaligned((&a as *const uint64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f64_f32(a: float32x2_t) -> float64x1_t {
    crate::ptr::read_unaligned((&a as *const float32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f64_p8(a: poly8x8_t) -> float64x1_t {
    crate::ptr::read_unaligned((&a as *const poly8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f64_p16(a: poly16x4_t) -> float64x1_t {
    crate::ptr::read_unaligned((&a as *const poly16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_f64_p64(a: poly64x1_t) -> float64x1_t {
    crate::ptr::read_unaligned((&a as *const poly64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p8_s8(a: int8x8_t) -> poly8x8_t {
    crate::ptr::read_unaligned((&a as *const int8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p8_s16(a: int16x4_t) -> poly8x8_t {
    crate::ptr::read_unaligned((&a as *const int16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p8_s32(a: int32x2_t) -> poly8x8_t {
    crate::ptr::read_unaligned((&a as *const int32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p8_s64(a: int64x1_t) -> poly8x8_t {
    crate::ptr::read_unaligned((&a as *const int64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p8_u8(a: uint8x8_t) -> poly8x8_t {
    crate::ptr::read_unaligned((&a as *const uint8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p8_u16(a: uint16x4_t) -> poly8x8_t {
    crate::ptr::read_unaligned((&a as *const uint16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p8_u32(a: uint32x2_t) -> poly8x8_t {
    crate::ptr::read_unaligned((&a as *const uint32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p8_u64(a: uint64x1_t) -> poly8x8_t {
    crate::ptr::read_unaligned((&a as *const uint64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p8_f32(a: float32x2_t) -> poly8x8_t {
    crate::ptr::read_unaligned((&a as *const float32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p8_f64(a: float64x1_t) -> poly8x8_t {
    crate::ptr::read_unaligned((&a as *const float64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p8_p16(a: poly16x4_t) -> poly8x8_t {
    crate::ptr::read_unaligned((&a as *const poly16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p8_p64(a: poly64x1_t) -> poly8x8_t {
    crate::ptr::read_unaligned((&a as *const poly64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p16_s8(a: int8x8_t) -> poly16x4_t {
    crate::ptr::read_unaligned((&a as *const int8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p16_s16(a: int16x4_t) -> poly16x4_t {
    crate::ptr::read_unaligned((&a as *const int16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p16_s32(a: int32x2_t) -> poly16x4_t {
    crate::ptr::read_unaligned((&a as *const int32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p16_s64(a: int64x1_t) -> poly16x4_t {
    crate::ptr::read_unaligned((&a as *const int64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p16_u8(a: uint8x8_t) -> poly16x4_t {
    crate::ptr::read_unaligned((&a as *const uint8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p16_u16(a: uint16x4_t) -> poly16x4_t {
    crate::ptr::read_unaligned((&a as *const uint16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p16_u32(a: uint32x2_t) -> poly16x4_t {
    crate::ptr::read_unaligned((&a as *const uint32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p16_u64(a: uint64x1_t) -> poly16x4_t {
    crate::ptr::read_unaligned((&a as *const uint64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p16_f32(a: float32x2_t) -> poly16x4_t {
    crate::ptr::read_unaligned((&a as *const float32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p16_f64(a: float64x1_t) -> poly16x4_t {
    crate::ptr::read_unaligned((&a as *const float64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p16_p8(a: poly8x8_t) -> poly16x4_t {
    crate::ptr::read_unaligned((&a as *const poly8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p16_p64(a: poly64x1_t) -> poly16x4_t {
    crate::ptr::read_unaligned((&a as *const poly64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p64_s8(a: int8x8_t) -> poly64x1_t {
    crate::ptr::read_unaligned((&a as *const int8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p64_s16(a: int16x4_t) -> poly64x1_t {
    crate::ptr::read_unaligned((&a as *const int16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p64_s32(a: int32x2_t) -> poly64x1_t {
    crate::ptr::read_unaligned((&a as *const int32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p64_s64(a: int64x1_t) -> poly64x1_t {
    crate::ptr::read_unaligned((&a as *const int64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p64_u8(a: uint8x8_t) -> poly64x1_t {
    crate::ptr::read_unaligned((&a as *const uint8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p64_u16(a: uint16x4_t) -> poly64x1_t {
    crate::ptr::read_unaligned((&a as *const uint16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p64_u32(a: uint32x2_t) -> poly64x1_t {
    crate::ptr::read_unaligned((&a as *const uint32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p64_u64(a: uint64x1_t) -> poly64x1_t {
    crate::ptr::read_unaligned((&a as *const uint64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p64_f32(a: float32x2_t) -> poly64x1_t {
    crate::ptr::read_unaligned((&a as *const float32x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p64_f64(a: float64x1_t) -> poly64x1_t {
    crate::ptr::read_unaligned((&a as *const float64x1_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p64_p8(a: poly8x8_t) -> poly64x1_t {
    crate::ptr::read_unaligned((&a as *const poly8x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpret_p64_p16(a: poly16x4_t) -> poly64x1_t {
    crate::ptr::read_unaligned((&a as *const poly16x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s8_s16(a: int16x8_t) -> int8x16_t {
    crate::ptr::read_unaligned((&a as *const int16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s8_s32(a: int32x4_t) -> int8x16_t {
    crate::ptr::read_unaligned((&a as *const int32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s8_s64(a: int64x2_t) -> int8x16_t {
    crate::ptr::read_unaligned((&a as *const int64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s8_u16(a: uint16x8_t) -> int8x16_t {
    crate::ptr::read_unaligned((&a as *const uint16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s8_u32(a: uint32x4_t) -> int8x16_t {
    crate::ptr::read_unaligned((&a as *const uint32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s8_u64(a: uint64x2_t) -> int8x16_t {
    crate::ptr::read_unaligned((&a as *const uint64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s8_f32(a: float32x4_t) -> int8x16_t {
    crate::ptr::read_unaligned((&a as *const float32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s8_f64(a: float64x2_t) -> int8x16_t {
    crate::ptr::read_unaligned((&a as *const float64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s8_p8(a: poly8x16_t) -> int8x16_t {
    crate::ptr::read_unaligned((&a as *const poly8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s8_p16(a: poly16x8_t) -> int8x16_t {
    crate::ptr::read_unaligned((&a as *const poly16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s8_p64(a: poly64x2_t) -> int8x16_t {
    crate::ptr::read_unaligned((&a as *const poly64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s8_p128(a: p128) -> int8x16_t {
    crate::ptr::read_unaligned((&a as *const p128).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s16_s8(a: int8x16_t) -> int16x8_t {
    crate::ptr::read_unaligned((&a as *const int8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s16_s64(a: int64x2_t) -> int16x8_t {
    crate::ptr::read_unaligned((&a as *const int64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s16_u8(a: uint8x16_t) -> int16x8_t {
    crate::ptr::read_unaligned((&a as *const uint8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s16_u32(a: uint32x4_t) -> int16x8_t {
    crate::ptr::read_unaligned((&a as *const uint32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s16_u64(a: uint64x2_t) -> int16x8_t {
    crate::ptr::read_unaligned((&a as *const uint64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s16_f32(a: float32x4_t) -> int16x8_t {
    crate::ptr::read_unaligned((&a as *const float32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s16_f64(a: float64x2_t) -> int16x8_t {
    crate::ptr::read_unaligned((&a as *const float64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s16_p8(a: poly8x16_t) -> int16x8_t {
    crate::ptr::read_unaligned((&a as *const poly8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s16_p16(a: poly16x8_t) -> int16x8_t {
    crate::ptr::read_unaligned((&a as *const poly16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s16_p64(a: poly64x2_t) -> int16x8_t {
    crate::ptr::read_unaligned((&a as *const poly64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s16_p128(a: p128) -> int16x8_t {
    crate::ptr::read_unaligned((&a as *const p128).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s32_s8(a: int8x16_t) -> int32x4_t {
    crate::ptr::read_unaligned((&a as *const int8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s32_u8(a: uint8x16_t) -> int32x4_t {
    crate::ptr::read_unaligned((&a as *const uint8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s32_u16(a: uint16x8_t) -> int32x4_t {
    crate::ptr::read_unaligned((&a as *const uint16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s32_u64(a: uint64x2_t) -> int32x4_t {
    crate::ptr::read_unaligned((&a as *const uint64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s32_f64(a: float64x2_t) -> int32x4_t {
    crate::ptr::read_unaligned((&a as *const float64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s32_p8(a: poly8x16_t) -> int32x4_t {
    crate::ptr::read_unaligned((&a as *const poly8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s32_p16(a: poly16x8_t) -> int32x4_t {
    crate::ptr::read_unaligned((&a as *const poly16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s32_p64(a: poly64x2_t) -> int32x4_t {
    crate::ptr::read_unaligned((&a as *const poly64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s32_p128(a: p128) -> int32x4_t {
    crate::ptr::read_unaligned((&a as *const p128).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s64_s8(a: int8x16_t) -> int64x2_t {
    crate::ptr::read_unaligned((&a as *const int8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s64_s16(a: int16x8_t) -> int64x2_t {
    crate::ptr::read_unaligned((&a as *const int16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s64_u8(a: uint8x16_t) -> int64x2_t {
    crate::ptr::read_unaligned((&a as *const uint8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s64_u16(a: uint16x8_t) -> int64x2_t {
    crate::ptr::read_unaligned((&a as *const uint16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s64_u32(a: uint32x4_t) -> int64x2_t {
    crate::ptr::read_unaligned((&a as *const uint32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s64_f32(a: float32x4_t) -> int64x2_t {
    crate::ptr::read_unaligned((&a as *const float32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s64_f64(a: float64x2_t) -> int64x2_t {
    crate::ptr::read_unaligned((&a as *const float64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s64_p8(a: poly8x16_t) -> int64x2_t {
    crate::ptr::read_unaligned((&a as *const poly8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s64_p16(a: poly16x8_t) -> int64x2_t {
    crate::ptr::read_unaligned((&a as *const poly16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s64_p64(a: poly64x2_t) -> int64x2_t {
    crate::ptr::read_unaligned((&a as *const poly64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_s64_p128(a: p128) -> int64x2_t {
    crate::ptr::read_unaligned((&a as *const p128).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u8_s16(a: int16x8_t) -> uint8x16_t {
    crate::ptr::read_unaligned((&a as *const int16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u8_s64(a: int64x2_t) -> uint8x16_t {
    crate::ptr::read_unaligned((&a as *const int64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u8_u16(a: uint16x8_t) -> uint8x16_t {
    crate::ptr::read_unaligned((&a as *const uint16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u8_u32(a: uint32x4_t) -> uint8x16_t {
    crate::ptr::read_unaligned((&a as *const uint32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u8_u64(a: uint64x2_t) -> uint8x16_t {
    crate::ptr::read_unaligned((&a as *const uint64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u8_f32(a: float32x4_t) -> uint8x16_t {
    crate::ptr::read_unaligned((&a as *const float32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u8_f64(a: float64x2_t) -> uint8x16_t {
    crate::ptr::read_unaligned((&a as *const float64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u8_p8(a: poly8x16_t) -> uint8x16_t {
    crate::ptr::read_unaligned((&a as *const poly8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u8_p16(a: poly16x8_t) -> uint8x16_t {
    crate::ptr::read_unaligned((&a as *const poly16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u8_p64(a: poly64x2_t) -> uint8x16_t {
    crate::ptr::read_unaligned((&a as *const poly64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u8_p128(a: p128) -> uint8x16_t {
    crate::ptr::read_unaligned((&a as *const p128).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u16_s8(a: int8x16_t) -> uint16x8_t {
    crate::ptr::read_unaligned((&a as *const int8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u16_s64(a: int64x2_t) -> uint16x8_t {
    crate::ptr::read_unaligned((&a as *const int64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u16_u64(a: uint64x2_t) -> uint16x8_t {
    crate::ptr::read_unaligned((&a as *const uint64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u16_f32(a: float32x4_t) -> uint16x8_t {
    crate::ptr::read_unaligned((&a as *const float32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u16_f64(a: float64x2_t) -> uint16x8_t {
    crate::ptr::read_unaligned((&a as *const float64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u16_p8(a: poly8x16_t) -> uint16x8_t {
    crate::ptr::read_unaligned((&a as *const poly8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u16_p16(a: poly16x8_t) -> uint16x8_t {
    crate::ptr::read_unaligned((&a as *const poly16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u16_p64(a: poly64x2_t) -> uint16x8_t {
    crate::ptr::read_unaligned((&a as *const poly64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u16_p128(a: p128) -> uint16x8_t {
    crate::ptr::read_unaligned((&a as *const p128).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u32_s8(a: int8x16_t) -> uint32x4_t {
    crate::ptr::read_unaligned((&a as *const int8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u32_s16(a: int16x8_t) -> uint32x4_t {
    crate::ptr::read_unaligned((&a as *const int16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u32_s64(a: int64x2_t) -> uint32x4_t {
    crate::ptr::read_unaligned((&a as *const int64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u32_u16(a: uint16x8_t) -> uint32x4_t {
    crate::ptr::read_unaligned((&a as *const uint16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u32_u64(a: uint64x2_t) -> uint32x4_t {
    crate::ptr::read_unaligned((&a as *const uint64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u32_p8(a: poly8x16_t) -> uint32x4_t {
    crate::ptr::read_unaligned((&a as *const poly8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u32_p16(a: poly16x8_t) -> uint32x4_t {
    crate::ptr::read_unaligned((&a as *const poly16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u32_p64(a: poly64x2_t) -> uint32x4_t {
    crate::ptr::read_unaligned((&a as *const poly64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u32_p128(a: p128) -> uint32x4_t {
    crate::ptr::read_unaligned((&a as *const p128).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u64_s8(a: int8x16_t) -> uint64x2_t {
    crate::ptr::read_unaligned((&a as *const int8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u64_s16(a: int16x8_t) -> uint64x2_t {
    crate::ptr::read_unaligned((&a as *const int16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u64_s32(a: int32x4_t) -> uint64x2_t {
    crate::ptr::read_unaligned((&a as *const int32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u64_u16(a: uint16x8_t) -> uint64x2_t {
    crate::ptr::read_unaligned((&a as *const uint16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u64_u32(a: uint32x4_t) -> uint64x2_t {
    crate::ptr::read_unaligned((&a as *const uint32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u64_p8(a: poly8x16_t) -> uint64x2_t {
    crate::ptr::read_unaligned((&a as *const poly8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u64_p16(a: poly16x8_t) -> uint64x2_t {
    crate::ptr::read_unaligned((&a as *const poly16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u64_p64(a: poly64x2_t) -> uint64x2_t {
    crate::ptr::read_unaligned((&a as *const poly64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_u64_p128(a: p128) -> uint64x2_t {
    crate::ptr::read_unaligned((&a as *const p128).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f32_s8(a: int8x16_t) -> float32x4_t {
    crate::ptr::read_unaligned((&a as *const int8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f32_s16(a: int16x8_t) -> float32x4_t {
    crate::ptr::read_unaligned((&a as *const int16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f32_s64(a: int64x2_t) -> float32x4_t {
    crate::ptr::read_unaligned((&a as *const int64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f32_u16(a: uint16x8_t) -> float32x4_t {
    crate::ptr::read_unaligned((&a as *const uint16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f32_f64(a: float64x2_t) -> float32x4_t {
    crate::ptr::read_unaligned((&a as *const float64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f32_p8(a: poly8x16_t) -> float32x4_t {
    crate::ptr::read_unaligned((&a as *const poly8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f32_p16(a: poly16x8_t) -> float32x4_t {
    crate::ptr::read_unaligned((&a as *const poly16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f32_p64(a: poly64x2_t) -> float32x4_t {
    crate::ptr::read_unaligned((&a as *const poly64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f32_p128(a: p128) -> float32x4_t {
    crate::ptr::read_unaligned((&a as *const p128).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f64_s8(a: int8x16_t) -> float64x2_t {
    crate::ptr::read_unaligned((&a as *const int8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f64_s16(a: int16x8_t) -> float64x2_t {
    crate::ptr::read_unaligned((&a as *const int16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f64_s32(a: int32x4_t) -> float64x2_t {
    crate::ptr::read_unaligned((&a as *const int32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f64_s64(a: int64x2_t) -> float64x2_t {
    crate::ptr::read_unaligned((&a as *const int64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f64_u8(a: uint8x16_t) -> float64x2_t {
    crate::ptr::read_unaligned((&a as *const uint8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f64_u16(a: uint16x8_t) -> float64x2_t {
    crate::ptr::read_unaligned((&a as *const uint16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f64_f32(a: float32x4_t) -> float64x2_t {
    crate::ptr::read_unaligned((&a as *const float32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f64_p8(a: poly8x16_t) -> float64x2_t {
    crate::ptr::read_unaligned((&a as *const poly8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f64_p16(a: poly16x8_t) -> float64x2_t {
    crate::ptr::read_unaligned((&a as *const poly16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f64_p64(a: poly64x2_t) -> float64x2_t {
    crate::ptr::read_unaligned((&a as *const poly64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_f64_p128(a: p128) -> float64x2_t {
    crate::ptr::read_unaligned((&a as *const p128).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p8_s8(a: int8x16_t) -> poly8x16_t {
    crate::ptr::read_unaligned((&a as *const int8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p8_s16(a: int16x8_t) -> poly8x16_t {
    crate::ptr::read_unaligned((&a as *const int16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p8_s32(a: int32x4_t) -> poly8x16_t {
    crate::ptr::read_unaligned((&a as *const int32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p8_s64(a: int64x2_t) -> poly8x16_t {
    crate::ptr::read_unaligned((&a as *const int64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p8_u8(a: uint8x16_t) -> poly8x16_t {
    crate::ptr::read_unaligned((&a as *const uint8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p8_u16(a: uint16x8_t) -> poly8x16_t {
    crate::ptr::read_unaligned((&a as *const uint16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p8_u32(a: uint32x4_t) -> poly8x16_t {
    crate::ptr::read_unaligned((&a as *const uint32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p8_u64(a: uint64x2_t) -> poly8x16_t {
    crate::ptr::read_unaligned((&a as *const uint64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p8_f32(a: float32x4_t) -> poly8x16_t {
    crate::ptr::read_unaligned((&a as *const float32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p8_f64(a: float64x2_t) -> poly8x16_t {
    crate::ptr::read_unaligned((&a as *const float64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p8_p16(a: poly16x8_t) -> poly8x16_t {
    crate::ptr::read_unaligned((&a as *const poly16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p8_p64(a: poly64x2_t) -> poly8x16_t {
    crate::ptr::read_unaligned((&a as *const poly64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p8_p128(a: p128) -> poly8x16_t {
    crate::ptr::read_unaligned((&a as *const p128).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p16_s8(a: int8x16_t) -> poly16x8_t {
    crate::ptr::read_unaligned((&a as *const int8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p16_s16(a: int16x8_t) -> poly16x8_t {
    crate::ptr::read_unaligned((&a as *const int16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p16_s32(a: int32x4_t) -> poly16x8_t {
    crate::ptr::read_unaligned((&a as *const int32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p16_s64(a: int64x2_t) -> poly16x8_t {
    crate::ptr::read_unaligned((&a as *const int64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p16_u8(a: uint8x16_t) -> poly16x8_t {
    crate::ptr::read_unaligned((&a as *const uint8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p16_u16(a: uint16x8_t) -> poly16x8_t {
    crate::ptr::read_unaligned((&a as *const uint16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p16_u32(a: uint32x4_t) -> poly16x8_t {
    crate::ptr::read_unaligned((&a as *const uint32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p16_u64(a: uint64x2_t) -> poly16x8_t {
    crate::ptr::read_unaligned((&a as *const uint64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p16_f32(a: float32x4_t) -> poly16x8_t {
    crate::ptr::read_unaligned((&a as *const float32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p16_f64(a: float64x2_t) -> poly16x8_t {
    crate::ptr::read_unaligned((&a as *const float64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p16_p8(a: poly8x16_t) -> poly16x8_t {
    crate::ptr::read_unaligned((&a as *const poly8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p16_p64(a: poly64x2_t) -> poly16x8_t {
    crate::ptr::read_unaligned((&a as *const poly64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p16_p128(a: p128) -> poly16x8_t {
    crate::ptr::read_unaligned((&a as *const p128).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p64_s8(a: int8x16_t) -> poly64x2_t {
    crate::ptr::read_unaligned((&a as *const int8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p64_s16(a: int16x8_t) -> poly64x2_t {
    crate::ptr::read_unaligned((&a as *const int16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p64_s32(a: int32x4_t) -> poly64x2_t {
    crate::ptr::read_unaligned((&a as *const int32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p64_s64(a: int64x2_t) -> poly64x2_t {
    crate::ptr::read_unaligned((&a as *const int64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p64_u8(a: uint8x16_t) -> poly64x2_t {
    crate::ptr::read_unaligned((&a as *const uint8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p64_u16(a: uint16x8_t) -> poly64x2_t {
    crate::ptr::read_unaligned((&a as *const uint16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p64_u32(a: uint32x4_t) -> poly64x2_t {
    crate::ptr::read_unaligned((&a as *const uint32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p64_u64(a: uint64x2_t) -> poly64x2_t {
    crate::ptr::read_unaligned((&a as *const uint64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p64_f32(a: float32x4_t) -> poly64x2_t {
    crate::ptr::read_unaligned((&a as *const float32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p64_f64(a: float64x2_t) -> poly64x2_t {
    crate::ptr::read_unaligned((&a as *const float64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p64_p8(a: poly8x16_t) -> poly64x2_t {
    crate::ptr::read_unaligned((&a as *const poly8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p64_p16(a: poly16x8_t) -> poly64x2_t {
    crate::ptr::read_unaligned((&a as *const poly16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p64_p128(a: p128) -> poly64x2_t {
    crate::ptr::read_unaligned((&a as *const p128).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p128_s8(a: int8x16_t) -> p128 {
    crate::ptr::read_unaligned((&a as *const int8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p128_s16(a: int16x8_t) -> p128 {
    crate::ptr::read_unaligned((&a as *const int16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p128_s32(a: int32x4_t) -> p128 {
    crate::ptr::read_unaligned((&a as *const int32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p128_s64(a: int64x2_t) -> p128 {
    crate::ptr::read_unaligned((&a as *const int64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p128_u8(a: uint8x16_t) -> p128 {
    crate::ptr::read_unaligned((&a as *const uint8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p128_u16(a: uint16x8_t) -> p128 {
    crate::ptr::read_unaligned((&a as *const uint16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p128_u32(a: uint32x4_t) -> p128 {
    crate::ptr::read_unaligned((&a as *const uint32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p128_u64(a: uint64x2_t) -> p128 {
    crate::ptr::read_unaligned((&a as *const uint64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p128_f32(a: float32x4_t) -> p128 {
    crate::ptr::read_unaligned((&a as *const float32x4_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p128_f64(a: float64x2_t) -> p128 {
    crate::ptr::read_unaligned((&a as *const float64x2_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p128_p8(a: poly8x16_t) -> p128 {
    crate::ptr::read_unaligned((&a as *const poly8x16_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p128_p16(a: poly16x8_t) -> p128 {
    crate::ptr::read_unaligned((&a as *const poly16x8_t).cast())
}

#[inline(always)]
pub unsafe fn vreinterpretq_p128_p64(a: poly64x2_t) -> p128 {
    crate::ptr::read_unaligned((&a as *const poly64x2_t).cast())
}
