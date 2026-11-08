//! Open string sets for APIs and providers.
//!
//! Dispatch upstream is by string, so these are newtypes over
//! `Cow<'static, str>` with associated constants for every value known at
//! v1.5.0 rather than closed enums.

use std::borrow::Cow;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

macro_rules! open_string {
    ($name:ident, $doc:literal, $( $const:ident = $val:literal ),+ $(,)?) => {
        #[doc = $doc]
        #[derive(Clone, PartialEq, Eq, Hash)]
        pub struct $name(Cow<'static, str>);

        impl $name {
            $( #[doc = concat!("Known value `", $val, "`.")] pub const $const: Self = Self(Cow::Borrowed($val)); )+

            /// The raw string value.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(Cow::Owned(value))
            }
        }

        impl From<&'static str> for $name {
            fn from(value: &'static str) -> Self {
                Self(Cow::Borrowed(value))
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({:?})", stringify!($name), self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(&self.0)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                String::deserialize(deserializer).map(Self::from)
            }
        }
    };
}

open_string!(
    Api,
    "Chat API family of a model or message (`KnownApi` is open).",
    ANTHROPIC_MESSAGES = "anthropic-messages",
    AZURE_OPENAI_RESPONSES = "azure-openai-responses",
    BEDROCK_CONVERSE_STREAM = "bedrock-converse-stream",
    GOOGLE_GENERATIVE_AI = "google-generative-ai",
    GOOGLE_VERTEX = "google-vertex",
    MISTRAL_CONVERSATIONS = "mistral-conversations",
    OPENAI_CODEX_RESPONSES = "openai-codex-responses",
    OPENAI_COMPLETIONS = "openai-completions",
    OPENAI_RESPONSES = "openai-responses",
    PI_MESSAGES = "pi-messages",
);

open_string!(
    ImageApi,
    "Image-generation API family (`KnownImageApi` is open).",
    OPENROUTER_IMAGES = "openrouter-images",
);

open_string!(
    ClassifierApi,
    "Structured classifier API family (`KnownClassifierApi` is open).",
    TYPESAFE_SYSTEM_ONE = "typesafe-system-one",
    CLOUDFLARE_WORKERS_AI_SYSTEM_ONE = "cloudflare-workers-ai-system-one",
    LLAMA_CPP_CLASSIFY = "llama-cpp-classify",
    OPENAI_DECISIONS = "openai-decisions",
);

open_string!(
    ProviderId,
    "Provider identifier (`KnownProvider` is open).",
    AMAZON_BEDROCK = "amazon-bedrock",
    ANT_LING = "ant-ling",
    ANTHROPIC = "anthropic",
    AZURE = "azure",
    BASETEN = "baseten",
    CEREBRAS = "cerebras",
    CLOUDFLARE_AI_GATEWAY = "cloudflare-ai-gateway",
    CLOUDFLARE_WORKERS_AI = "cloudflare-workers-ai",
    DEEPSEEK = "deepseek",
    FIREWORKS = "fireworks",
    GITHUB_COPILOT = "github-copilot",
    GOOGLE = "google",
    GOOGLE_VERTEX = "google-vertex",
    GROQ = "groq",
    HUGGINGFACE = "huggingface",
    KIMI_CODING = "kimi-coding",
    META = "meta",
    MINIMAX = "minimax",
    MINIMAX_CN = "minimax-cn",
    MISTRAL = "mistral",
    MOONSHOTAI = "moonshotai",
    MOONSHOTAI_CN = "moonshotai-cn",
    NVIDIA = "nvidia",
    OPENCODE = "opencode",
    OPENCODE_GO = "opencode-go",
    OPENAI = "openai",
    OPENAI_CODEX = "openai-codex",
    OPENROUTER = "openrouter",
    QWEN_TOKEN_PLAN = "qwen-token-plan",
    QWEN_TOKEN_PLAN_CN = "qwen-token-plan-cn",
    QWEN_TOKEN_PLAN_INDIVIDUAL = "qwen-token-plan-individual",
    RADIUS = "radius",
    TOGETHER = "together",
    TYPESAFE = "typesafe",
    VERCEL_AI_GATEWAY = "vercel-ai-gateway",
    XAI = "xai",
    XIAOMI = "xiaomi",
    XIAOMI_TOKEN_PLAN_AMS = "xiaomi-token-plan-ams",
    XIAOMI_TOKEN_PLAN_CN = "xiaomi-token-plan-cn",
    XIAOMI_TOKEN_PLAN_SGP = "xiaomi-token-plan-sgp",
    ZAI = "zai",
    ZAI_CODING_CN = "zai-coding-cn",
);
