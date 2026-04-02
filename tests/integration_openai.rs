//! Integration tests against the real OpenAI API.
//! Run with: OPENAI_API_KEY=... cargo test --test integration_openai -- --ignored

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use yoagent::agent_loop::{agent_loop, AgentLoopConfig};
use yoagent::provider::{ModelConfig, OpenAiCompatProvider};
use yoagent::tools::WebSearchTool;
use yoagent::types::*;

fn api_key() -> String {
    std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set")
}

fn make_config() -> AgentLoopConfig {
    let model_config = ModelConfig::openai("gpt-4o-search-preview", "GPT-4o Search");
    AgentLoopConfig {
        provider: std::sync::Arc::new(OpenAiCompatProvider),
        model: "gpt-4o-search-preview".into(),
        api_key: api_key(),
        thinking_level: ThinkingLevel::Off,
        max_tokens: Some(1024),
        temperature: None,
        model_config: Some(model_config),
        convert_to_llm: None,
        transform_context: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        context_config: None,
        compaction_strategy: None,
        execution_limits: None,
        cache_config: CacheConfig {
            enabled: false,
            strategy: CacheStrategy::Disabled,
        },
        tool_execution: ToolExecutionStrategy::default(),
        retry_config: yoagent::RetryConfig::default(),
        before_turn: None,
        after_turn: None,
        on_error: None,
        input_filters: vec![],
    }
}

fn extract_assistant_text(messages: &[AgentMessage]) -> String {
    messages
        .iter()
        .filter_map(|m| {
            if let AgentMessage::Llm(Message::Assistant { content, .. }) = m {
                Some(
                    content
                        .iter()
                        .filter_map(|c| {
                            if let Content::Text { text } = c {
                                Some(text.as_str())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                )
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Web search with gpt-4o-search-preview.
#[tokio::test]
#[ignore]
async fn test_openai_web_search() {
    let config = make_config();
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let mut context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: vec![Box::new(WebSearchTool)],
    };

    let prompt = AgentMessage::Llm(Message::user("What is the current population of Toronto?"));
    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    let text = extract_assistant_text(&new_messages);
    assert!(!text.is_empty(), "Expected non-empty response");
    assert!(
        text.contains("Toronto") || text.contains("population") || text.contains("million"),
        "Expected response about Toronto's population, got: {}",
        text
    );
    println!("Response: {}", text);
}
