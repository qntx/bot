//! One-pass scanner that repairs lone `UTF-16` surrogate `\u` escapes so
//! `serde_json` can parse input that `JSON.parse` accepts.

/// Return a copy of `text` with every lone surrogate escape replaced by
/// `\ufffd`, or `None` when nothing needed repair. Escapes other than
/// `\uXXXX` (`\\`, `\"`, `\n`, ...) are stepped over so `\\ud800` (escaped
/// backslash followed by text) is untouched.
pub(crate) fn repair_lone_surrogates(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut repaired: Option<Vec<u8>> = None;
    let mut kept = 0usize;
    let mut i = 0usize;
    while i + 1 < bytes.len() {
        i += match bytes.get(i..i + 2) {
            Some(b"\\\\") => 2,
            Some(b"\\u") => match classify_escape(bytes, i) {
                Escape::Lone => {
                    replace(&mut repaired, bytes, &mut kept, i);
                    6
                }
                Escape::Pair => 12,
                Escape::Plain => 6,
                // Invalid digits: step over only `\u` so a backslash inside
                // the would-be digits is still stepped over.
                Escape::Short => 2,
            },
            _ => 1,
        };
    }
    // Splices happen only at ASCII escape boundaries, so the bytes stay UTF-8.
    repaired.and_then(|mut out| {
        out.extend_from_slice(bytes.get(kept..)?);
        String::from_utf8(out).ok()
    })
}

/// What the `\u` escape at `index` is.
enum Escape {
    /// A surrogate escape without its pair partner: repaired to `\ufffd`.
    Lone,
    /// A valid high+low surrogate pair.
    Pair,
    /// A `\uXXXX` escape outside the surrogate range.
    Plain,
    /// `\u` not followed by four hex digits.
    Short,
}

fn classify_escape(bytes: &[u8], index: usize) -> Escape {
    match bytes.get(index + 2..index + 6).and_then(hex4) {
        Some(0xD800..=0xDBFF) => {
            if paired_low(bytes, index + 6) {
                Escape::Pair
            } else {
                Escape::Lone
            }
        }
        Some(0xDC00..=0xDFFF) => Escape::Lone,
        Some(_) => Escape::Plain,
        None => Escape::Short,
    }
}

/// Whether the escape at `index` is `\uDC00`–`\uDFFF`.
fn paired_low(bytes: &[u8], index: usize) -> bool {
    bytes.get(index..index + 2) == Some(&b"\\u"[..])
        && matches!(
            bytes.get(index + 2..index + 6).and_then(hex4),
            Some(0xDC00..=0xDFFF)
        )
}

/// Append the unchanged span before the escape at `index`, then `\ufffd`.
fn replace(repaired: &mut Option<Vec<u8>>, bytes: &[u8], kept: &mut usize, index: usize) {
    let out = repaired.get_or_insert_with(Vec::new);
    if let Some(span) = bytes.get(*kept..index) {
        out.extend_from_slice(span);
    }
    out.extend_from_slice(b"\\ufffd");
    *kept = index + 6;
}

fn hex4(digits: &[u8]) -> Option<u16> {
    if digits.len() != 4 {
        return None;
    }
    digits.iter().try_fold(0u16, |value, &byte| {
        let digit = char::from(byte).to_digit(16)?;
        Some(value * 16 + u16::try_from(digit).ok()?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repairs_lone_high_and_low_surrogates() {
        assert_eq!(
            repair_lone_surrogates(r#""\ud800x""#).as_deref(),
            Some(r#""\ufffdx""#)
        );
        assert_eq!(
            repair_lone_surrogates(r#""\udc00""#).as_deref(),
            Some(r#""\ufffd""#)
        );
    }

    #[test]
    fn keeps_paired_surrogates() {
        assert_eq!(repair_lone_surrogates(r#""\ud800\udc00""#), None);
        assert_eq!(repair_lone_surrogates(r#""ok""#), None);
    }

    #[test]
    fn steps_over_other_escapes() {
        assert_eq!(repair_lone_surrogates(r#""\\ud800""#), None);
        assert_eq!(
            repair_lone_surrogates(r#""a\n\udc00b""#).as_deref(),
            Some(r#""a\n\ufffdb""#)
        );
    }

    #[test]
    fn repairs_multiple_lone_surrogates() {
        assert_eq!(
            repair_lone_surrogates(r#""\ud800\ud800""#).as_deref(),
            Some(r#""\ufffd\ufffd""#)
        );
        assert_eq!(
            repair_lone_surrogates(r#""\ud800\ud800\udc00""#).as_deref(),
            Some(r#""\ufffd\ud800\udc00""#),
        );
    }
}
