//! Shared helpers for integration tests.

/// Compare two JSON values with JS semantics: objects order-insensitive,
/// arrays ordered, numbers equal when their `f64` values are equal.
pub(crate) fn js_value_eq(left: &serde_json::Value, right: &serde_json::Value) -> bool {
    use serde_json::Value::*;
    match (left, right) {
        (Number(a), Number(b)) => a.as_f64() == b.as_f64(),
        (Object(a), Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, value)| b.get(key).is_some_and(|other| js_value_eq(value, other)))
        }
        (Array(a), Array(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| js_value_eq(x, y))
        }
        _ => left == right,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn compares_numbers_by_f64() {
        assert!(js_value_eq(&json!(1), &json!(1.0)));
        assert!(!js_value_eq(&json!(1), &json!(2)));
    }

    #[test]
    fn compares_objects_order_insensitive() {
        assert!(js_value_eq(
            &json!({"a": 1, "b": {"c": 2}}),
            &json!({"b": {"c": 2}, "a": 1})
        ));
        assert!(!js_value_eq(&json!({"a": 1}), &json!({"a": 1, "b": 2})));
        assert!(!js_value_eq(&json!({"a": 1}), &json!({"b": 1})));
    }

    #[test]
    fn compares_arrays_ordered() {
        assert!(js_value_eq(&json!([1, 2]), &json!([1, 2])));
        assert!(!js_value_eq(&json!([1, 2]), &json!([2, 1])));
    }
}
