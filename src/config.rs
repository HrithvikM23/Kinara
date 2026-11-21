//! Pipeline configuration management and JSON persistence.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Global pipeline configuration parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub input_source: String,
    pub output_path: String,
    pub fps: f64,
    pub execution_backend: String,
    pub body_ik_enabled: bool,
    pub hand_ik_enabled: bool,
    pub foot_lock_enabled: bool,
    pub filter_min_cutoff: f32,
    pub filter_beta: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            input_source: "synthetic:30:640x480".to_string(),
            output_path: "output.fbx".to_string(),
            fps: 30.0,
            execution_backend: "cpu".to_string(),
            body_ik_enabled: true,
            hand_ik_enabled: true,
            foot_lock_enabled: true,
            filter_min_cutoff: 1.0,
            filter_beta: 0.007,
        }
    }
}

impl PipelineConfig {
    /// Loads configuration from JSON file or returns production defaults.
    pub fn load_or_default(path: &Path) -> Self {
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(cfg) = serde_json::from_str(&content) {
                    return cfg;
                }
            }
        }
        Self::default()
    }

    /// Persists configuration parameters to disk.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let serialized = serde_json::to_string_pretty(self)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_serialization() {
        let cfg = PipelineConfig::default();
        let serialized = serde_json::to_string(&cfg).expect("Failed to serialize");
        let deserialized: PipelineConfig = serde_json::from_str(&serialized).expect("Failed to deserialize");
        assert_eq!(cfg, deserialized);
    }
}
