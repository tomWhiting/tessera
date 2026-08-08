use super::ResourcePolicyError;

pub(super) fn parse_parameter_count(value: &str) -> Result<u128, ResourcePolicyError> {
    let registry_value = value;
    let value = value.trim();
    let suffix = value
        .chars()
        .last()
        .ok_or_else(|| invalid_parameter_count(registry_value))?;
    let multiplier = match suffix.to_ascii_uppercase() {
        'K' => 1_000_u128,
        'M' => 1_000_000_u128,
        'B' => 1_000_000_000_u128,
        _ if suffix.is_ascii_digit() => 1_u128,
        _ => return Err(invalid_parameter_count(registry_value)),
    };
    let magnitude = if multiplier == 1 {
        value
    } else {
        &value[..value.len() - suffix.len_utf8()]
    };

    let mut parts = magnitude.split('.');
    let whole = parts
        .next()
        .filter(|part| !part.is_empty())
        .ok_or_else(|| invalid_parameter_count(registry_value))?;
    let fraction = parts.next();
    if parts.next().is_some()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || fraction
            .is_some_and(|part| part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return Err(invalid_parameter_count(registry_value));
    }

    let whole = whole
        .parse::<u128>()
        .map_err(|_| invalid_parameter_count(registry_value))?;
    let (fraction, scale) = parse_fraction(fraction, registry_value)?;
    let scaled_magnitude = whole
        .checked_mul(scale)
        .and_then(|scaled| scaled.checked_add(fraction))
        .ok_or_else(|| invalid_parameter_count(registry_value))?;
    let scaled_parameters = scaled_magnitude
        .checked_mul(multiplier)
        .ok_or_else(|| invalid_parameter_count(registry_value))?;
    let rounded_up = u128::from(scaled_parameters % scale != 0);

    Ok(scaled_parameters / scale + rounded_up)
}

fn parse_fraction(
    fraction: Option<&str>,
    registry_value: &str,
) -> Result<(u128, u128), ResourcePolicyError> {
    let Some(fraction_text) = fraction else {
        return Ok((0, 1));
    };
    let fractional_digits =
        u32::try_from(fraction_text.len()).map_err(|_| invalid_parameter_count(registry_value))?;
    let fraction = fraction_text
        .parse::<u128>()
        .map_err(|_| invalid_parameter_count(registry_value))?;
    let scale = 10_u128
        .checked_pow(fractional_digits)
        .ok_or_else(|| invalid_parameter_count(registry_value))?;
    Ok((fraction, scale))
}

fn invalid_parameter_count(value: &str) -> ResourcePolicyError {
    ResourcePolicyError::InvalidParameterCount {
        value: value.to_string(),
    }
}
