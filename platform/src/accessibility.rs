//! Process-wide observed accessibility preferences for every hosted renderer.
//! Platform hosts update these from native readback; widgets never write OS settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccessibilityPreferences {
    pub high_contrast_text: bool,
    pub font_weight_adjustment: i32,
    animator_scale: f64,
}
impl Default for AccessibilityPreferences {
    fn default() -> Self {
        Self { high_contrast_text: false, font_weight_adjustment: 0, animator_scale: 1.0 }
    }
}
impl AccessibilityPreferences {
    pub fn observed(high_contrast_text: bool, font_weight_adjustment: i32, animator_scale: f64) -> Option<Self> {
        if !(-1000..=1000).contains(&font_weight_adjustment) || !animator_scale.is_finite() || animator_scale < 0.0 {
            return None;
        }
        Some(Self { high_contrast_text, font_weight_adjustment, animator_scale })
    }
    pub fn animator_scale(self) -> f64 {
        self.animator_scale
    }
    pub fn reduce_motion(self) -> bool {
        self.animator_scale == 0.0
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_preferences_preserve_custom_values_and_reject_unknowns() {
        assert_eq!(AccessibilityPreferences::default().animator_scale(), 1.0);
        let custom = AccessibilityPreferences::observed(true, 100, 0.5).unwrap();
        assert_eq!(custom.font_weight_adjustment, 100);
        assert!(!custom.reduce_motion());
        assert!(AccessibilityPreferences::observed(false, 0, 0.0).unwrap().reduce_motion());
        for scale in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(AccessibilityPreferences::observed(false, 0, scale).is_none());
        }
        assert!(AccessibilityPreferences::observed(false, i32::MAX, 1.0).is_none());
    }
}
