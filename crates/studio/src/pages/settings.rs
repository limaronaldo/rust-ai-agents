//! Settings page

use leptos::prelude::*;

/// Application settings stored in localStorage
#[derive(Debug, Clone, PartialEq)]
pub struct AppSettings {
    pub api_endpoint: String,
    pub ws_endpoint: String,
    pub theme: Theme,
    pub auto_connect: bool,
    pub refresh_interval_ms: u32,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            api_endpoint: String::new(), // Use current origin
            ws_endpoint: String::new(),  // Use current origin
            theme: Theme::Dark,
            auto_connect: true,
            refresh_interval_ms: 5000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Theme {
    Dark,
    Light,
    System,
}

impl Theme {
    pub fn as_str(&self) -> &'static str {
        match self {
            Theme::Dark => "dark",
            Theme::Light => "light",
            Theme::System => "system",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "light" => Theme::Light,
            "system" => Theme::System,
            _ => Theme::Dark,
        }
    }
}

/// Settings page component
#[component]
pub fn SettingsPage() -> impl IntoView {
    // Load settings from localStorage or use defaults
    let settings = RwSignal::new(load_settings());
    let saved = RwSignal::new(false);

    let save_settings = move |_| {
        let s = settings.get();
        save_settings_to_storage(&s);
        saved.set(true);
        // Reset saved indicator after 2 seconds
        set_timeout(move || saved.set(false), std::time::Duration::from_secs(2));
    };

    let reset_settings = move |_| {
        settings.set(AppSettings::default());
    };

    view! {
        <div>
            <div class="flex items-center justify-between mb-6">
                <h1 class="text-2xl font-bold text-white">"Settings"</h1>
                {move || saved.get().then(|| view! {
                    <span class="text-green-400 text-sm">"✓ Settings saved"</span>
                })}
            </div>

            <div class="space-y-6">
                // Connection Settings
                <SettingsSection title="Connection">
                    <SettingsField label="API Endpoint" description="Leave empty to use current origin">
                        <input
                            type="text"
                            placeholder="https://api.example.com"
                            class="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                            prop:value=move || settings.get().api_endpoint
                            on:input=move |e| {
                                settings.update(|s| s.api_endpoint = event_target_value(&e));
                            }
                        />
                    </SettingsField>

                    <SettingsField label="WebSocket Endpoint" description="Leave empty to use current origin">
                        <input
                            type="text"
                            placeholder="wss://api.example.com/ws"
                            class="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                            prop:value=move || settings.get().ws_endpoint
                            on:input=move |e| {
                                settings.update(|s| s.ws_endpoint = event_target_value(&e));
                            }
                        />
                    </SettingsField>

                    <SettingsField label="Auto-connect" description="Automatically connect WebSocket on page load">
                        <label class="flex items-center gap-2 cursor-pointer">
                            <input
                                type="checkbox"
                                class="w-5 h-5 rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500"
                                prop:checked=move || settings.get().auto_connect
                                on:change=move |e| {
                                    let checked = event_target_checked(&e);
                                    settings.update(|s| s.auto_connect = checked);
                                }
                            />
                            <span class="text-gray-300">"Enable auto-connect"</span>
                        </label>
                    </SettingsField>
                </SettingsSection>

                // Display Settings
                <SettingsSection title="Display">
                    <SettingsField label="Theme" description="Choose your preferred color theme">
                        <select
                            class="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                            on:change=move |e| {
                                let value = event_target_value(&e);
                                settings.update(|s| s.theme = Theme::from_str(&value));
                            }
                        >
                            <option value="dark" selected=move || settings.get().theme == Theme::Dark>"Dark"</option>
                            <option value="light" selected=move || settings.get().theme == Theme::Light>"Light"</option>
                            <option value="system" selected=move || settings.get().theme == Theme::System>"System"</option>
                        </select>
                    </SettingsField>

                    <SettingsField label="Refresh Interval" description="How often to refresh data (in milliseconds)">
                        <input
                            type="number"
                            min="1000"
                            max="60000"
                            step="1000"
                            class="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                            prop:value=move || settings.get().refresh_interval_ms
                            on:input=move |e| {
                                if let Ok(val) = event_target_value(&e).parse() {
                                    settings.update(|s| s.refresh_interval_ms = val);
                                }
                            }
                        />
                    </SettingsField>
                </SettingsSection>

                // Actions
                <div class="flex items-center gap-4 pt-4">
                    <button
                        class="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                        on:click=save_settings
                    >
                        "Save Settings"
                    </button>
                    <button
                        class="px-6 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-lg transition-colors"
                        on:click=reset_settings
                    >
                        "Reset to Defaults"
                    </button>
                </div>

                // About section
                <SettingsSection title="About">
                    <div class="text-gray-400 space-y-2">
                        <p>"Agent Studio v0.1.0"</p>
                        <p>"Built with Leptos 0.8 and Rust"</p>
                        <p>
                            <a href="https://github.com/limaronaldo/rust-ai-agents" target="_blank" class="text-blue-400 hover:text-blue-300">
                                "View on GitHub →"
                            </a>
                        </p>
                    </div>
                </SettingsSection>
            </div>
        </div>
    }
}

/// Settings section with title
#[component]
fn SettingsSection(title: &'static str, children: Children) -> impl IntoView {
    view! {
        <div class="bg-gray-800 rounded-lg p-6">
            <h2 class="text-lg font-semibold text-white mb-4">{title}</h2>
            <div class="space-y-4">
                {children()}
            </div>
        </div>
    }
}

/// Settings field with label and description
#[component]
fn SettingsField(
    label: &'static str,
    description: &'static str,
    children: Children,
) -> impl IntoView {
    view! {
        <div>
            <label class="block text-sm font-medium text-gray-300 mb-1">{label}</label>
            <p class="text-xs text-gray-500 mb-2">{description}</p>
            {children()}
        </div>
    }
}

/// Load settings from localStorage
fn load_settings() -> AppSettings {
    let window = web_sys::window().expect("no window");
    let storage = window.local_storage().ok().flatten();

    if let Some(storage) = storage {
        if let Ok(Some(json)) = storage.get_item("agent_studio_settings") {
            if let Ok(settings) = serde_json::from_str::<serde_json::Value>(&json) {
                return AppSettings {
                    api_endpoint: settings["api_endpoint"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    ws_endpoint: settings["ws_endpoint"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    theme: Theme::from_str(settings["theme"].as_str().unwrap_or("dark")),
                    auto_connect: settings["auto_connect"].as_bool().unwrap_or(true),
                    refresh_interval_ms: settings["refresh_interval_ms"].as_u64().unwrap_or(5000)
                        as u32,
                };
            }
        }
    }

    AppSettings::default()
}

/// Save settings to localStorage
fn save_settings_to_storage(settings: &AppSettings) {
    let window = web_sys::window().expect("no window");
    if let Ok(Some(storage)) = window.local_storage() {
        let json = serde_json::json!({
            "api_endpoint": settings.api_endpoint,
            "ws_endpoint": settings.ws_endpoint,
            "theme": settings.theme.as_str(),
            "auto_connect": settings.auto_connect,
            "refresh_interval_ms": settings.refresh_interval_ms,
        });
        let _ = storage.set_item("agent_studio_settings", &json.to_string());
    }
}
