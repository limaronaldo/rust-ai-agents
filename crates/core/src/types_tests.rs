//! Unit tests for core types

#[cfg(test)]
mod tests {
    use crate::types::*;

    #[test]
    fn test_agent_id_creation() {
        let id1 = AgentId::new("test-agent");
        assert_eq!(id1.0, "test-agent");
        assert_eq!(id1.to_string(), "test-agent");

        let id2 = AgentId::generate();
        assert!(!id2.0.is_empty());
        assert!(uuid::Uuid::parse_str(&id2.0).is_ok());
    }

    #[test]
    fn test_agent_id_equality() {
        let id1 = AgentId::new("agent-1");
        let id2 = AgentId::new("agent-1");
        let id3 = AgentId::new("agent-2");

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_agent_id_hashing() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        let id = AgentId::new("test");
        map.insert(id.clone(), "value");

        assert_eq!(map.get(&id), Some(&"value"));
    }

    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::new(1024);
        assert_eq!(config.max_size, 1024);
        assert_eq!(config.persist, false);
        assert!(matches!(
            config.retention_policy,
            RetentionPolicy::KeepRecent(100)
        ));
    }

    #[test]
    fn test_memory_config_builder() {
        let config = MemoryConfig::new(2048)
            .with_persistence(true)
            .with_retention(RetentionPolicy::KeepAll);

        assert_eq!(config.max_size, 2048);
        assert_eq!(config.persist, true);
        assert!(matches!(config.retention_policy, RetentionPolicy::KeepAll));
    }

    #[test]
    fn test_planning_mode_is_enabled() {
        assert!(!PlanningMode::Disabled.is_enabled());
        assert!(PlanningMode::BeforeTask.is_enabled());
        assert!(PlanningMode::FullPlan.is_enabled());
        assert!(PlanningMode::Adaptive.is_enabled());
    }

    #[test]
    fn test_planning_mode_default() {
        let mode: PlanningMode = Default::default();
        assert!(matches!(mode, PlanningMode::Disabled));
    }

    #[test]
    fn test_plan_step_creation() {
        let step = PlanStep::new(1, "Execute task");
        assert_eq!(step.step_number, 1);
        assert_eq!(step.description, "Execute task");
        assert_eq!(step.completed, false);
        assert!(step.actual_result.is_none());
        assert!(step.tools.is_empty());
    }

    #[test]
    fn test_plan_step_builder() {
        let step = PlanStep::new(1, "Analyze data")
            .with_expected_result("Analysis complete")
            .with_tools(vec!["analyzer".to_string(), "reporter".to_string()]);

        assert_eq!(step.expected_result, "Analysis complete");
        assert_eq!(step.tools.len(), 2);
        assert_eq!(step.tools[0], "analyzer");
    }

    #[test]
    fn test_plan_step_mark_completed() {
        let mut step = PlanStep::new(1, "Test");
        assert_eq!(step.completed, false);

        step.mark_completed("Success");
        assert_eq!(step.completed, true);
        assert_eq!(step.actual_result, Some("Success".to_string()));
    }

    #[test]
    fn test_capability_equality() {
        let cap1 = Capability::Analysis;
        let cap2 = Capability::Analysis;
        let cap3 = Capability::RiskAssessment;

        assert_eq!(cap1, cap2);
        assert_ne!(cap1, cap3);
    }

    #[test]
    fn test_capability_custom() {
        let cap = Capability::Custom("CustomCapability".to_string());
        assert!(matches!(cap, Capability::Custom(_)));
    }

    #[test]
    fn test_agent_role_variants() {
        let roles = vec![
            AgentRole::Researcher,
            AgentRole::Writer,
            AgentRole::Reviewer,
            AgentRole::Coordinator,
            AgentRole::Executor,
            AgentRole::Custom("Specialist".to_string()),
        ];

        assert_eq!(roles.len(), 6);
    }

    #[test]
    fn test_routing_strategy_variants() {
        let strategies = vec![
            RoutingStrategy::Direct,
            RoutingStrategy::Broadcast,
            RoutingStrategy::RoundRobin,
            RoutingStrategy::LoadBalanced,
            RoutingStrategy::Priority(5),
        ];

        assert_eq!(strategies.len(), 5);
    }

    #[test]
    fn test_retention_policy_variants() {
        let policies = vec![
            RetentionPolicy::KeepAll,
            RetentionPolicy::KeepRecent(50),
            RetentionPolicy::KeepImportant(0.8),
            RetentionPolicy::Custom,
        ];

        assert_eq!(policies.len(), 4);
    }

    #[test]
    fn test_serialization_agent_id() {
        let id = AgentId::new("test-agent");
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: AgentId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn test_serialization_memory_config() {
        let config = MemoryConfig::new(1024)
            .with_persistence(true)
            .with_retention(RetentionPolicy::KeepRecent(100));

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: MemoryConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.max_size, deserialized.max_size);
        assert_eq!(config.persist, deserialized.persist);
    }

    #[test]
    fn test_serialization_planning_mode() {
        let modes = vec![
            PlanningMode::Disabled,
            PlanningMode::BeforeTask,
            PlanningMode::FullPlan,
            PlanningMode::Adaptive,
        ];

        for mode in modes {
            let json = serde_json::to_string(&mode).unwrap();
            let deserialized: PlanningMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, deserialized);
        }
    }
}
