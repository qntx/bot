//! `utils/hash.ts` — the deterministic cyrb53-style `shortHash`.
//!
//! The hash iterates **UTF-16 code units** (JS `charCodeAt`), so non-BMP
//! input hashes differently than a Rust `chars()` walk — surrogate pairs
//! produce two code units like JS.

/// `shortHash` — `(h2>>>0).toString(36) + (h1>>>0).toString(36)`.
#[must_use]
pub fn short_hash(text: &str) -> String {
    let mut h1: u32 = 0xdead_beef;
    let mut h2: u32 = 0x41c6_ce57;
    for ch in text.encode_utf16() {
        h1 = (h1 ^ u32::from(ch)).wrapping_mul(2_654_435_761);
        h2 = (h2 ^ u32::from(ch)).wrapping_mul(1_597_334_677);
    }
    h1 = (h1 ^ (h1 >> 16)).wrapping_mul(2_246_822_507)
        ^ (h2 ^ (h2 >> 13)).wrapping_mul(3_266_489_909);
    h2 = (h2 ^ (h2 >> 16)).wrapping_mul(2_246_822_507)
        ^ (h1 ^ (h1 >> 13)).wrapping_mul(3_266_489_909);
    format!("{}{}", radix36(h2), radix36(h1))
}

/// `u32::to_string(36)` — JS `(n >>> 0).toString(36)`.
fn radix36(mut n: u32) -> String {
    if n == 0 {
        return String::from("0");
    }
    let mut out = Vec::new();
    while n > 0 {
        let digit = char::from_digit(n % 36, 36).unwrap_or('0');
        out.push(digit);
        n /= 36;
    }
    out.iter().rev().collect()
}
