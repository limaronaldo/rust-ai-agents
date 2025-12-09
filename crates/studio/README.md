# rust-ai-agents-studio

Web UI for monitoring and debugging AI agents, built with Leptos 0.8.

## Features

- **Dashboard** - Real-time metrics, cost tracking, request stats
- **Trace Viewer** - Monitor agent executions with detailed timelines
- **Session Browser** - View and manage conversation sessions
- **Agent Control** - Start, stop, and restart agents
- **Metrics Charts** - SVG charts for latency, tokens, costs
- **Chat UI** - Interactive chat with SSE streaming
- **Settings** - Configure connection and display options
- **Live Updates** - WebSocket connection for real-time data

## Installation

The Studio is a WASM application served by the dashboard backend.

```toml
[dependencies]
rust-ai-agents-studio = "0.1"
```

## Building

### Prerequisites

```bash
# Install trunk (WASM bundler)
cargo install trunk

# Add WASM target
rustup target add wasm32-unknown-unknown
```

### Build Commands

```bash
cd crates/studio

# Development build with hot reload
trunk serve

# Production build
trunk build --release

# Build to specific output
trunk build --release -d ../dashboard/static/studio
```

### Build Script

```bash
# Use the included build script
./build.sh
```

## Pages

| Route | Description |
|-------|-------------|
| `/` | Dashboard with metrics overview |
| `/traces` | Trace list with filtering |
| `/traces/:id` | Trace detail view |
| `/sessions` | Session list |
| `/sessions/:id` | Session messages and metadata |
| `/agents` | Agent list with controls |
| `/metrics` | Detailed metrics with charts |
| `/chat` | Interactive chat interface |
| `/settings` | Application settings |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Agent Studio                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Router                            │   │
│  │  /  /traces  /sessions  /agents  /metrics  /chat    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Components                         │   │
│  │  Layout | Navbar | Sidebar | Cards | Charts         │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  API Layer                           │   │
│  │  ApiClient (REST) | WsClient (WebSocket)            │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
└───────────────────────────┼─────────────────────────────────┘
                            │
                    ┌───────┴───────┐
                    │   Dashboard   │
                    │    (Axum)     │
                    └───────────────┘
```

## Components

### Layout

```rust
use rust_ai_agents_studio::components::Layout;

view! {
    <Layout>
        <YourContent />
    </Layout>
}
```

### Charts

```rust
use rust_ai_agents_studio::components::charts::{
    LineChart, BarChart, SparklineCard, Gauge
};

view! {
    // Line chart
    <LineChart
        data=data_signal
        width=400
        height=200
        title="Requests per Minute"
    />

    // Bar chart
    <BarChart
        data=bar_data
        width=300
        height=150
    />

    // Sparkline in a card
    <SparklineCard
        title="Latency"
        value="125ms"
        data=sparkline_data
        trend="up"
    />

    // Gauge for percentages
    <Gauge
        value=0.75
        label="CPU Usage"
    />
}
```

### Cards

```rust
use rust_ai_agents_studio::components::cards::{StatCard, MetricCard};

view! {
    <StatCard
        title="Total Requests"
        value="1,523"
        change="+12%"
        trend="up"
    />

    <MetricCard
        title="Token Usage"
        primary="245K"
        secondary="input: 180K, output: 65K"
    />
}
```

### Tables

```rust
use rust_ai_agents_studio::components::tables::DataTable;

view! {
    <DataTable
        headers=vec!["ID", "Status", "Duration"]
        rows=traces_signal
        on_row_click=|id| navigate_to_trace(id)
    />
}
```

## API Client

### REST Calls

```rust
use rust_ai_agents_studio::api::ApiClient;

let client = ApiClient::from_origin();

// Fetch metrics
let metrics = client.get_metrics().await?;

// Fetch sessions
let sessions = client.get_sessions().await?;

// Control agents
client.start_agent("agent-1").await?;
client.stop_agent("agent-1").await?;
```

### WebSocket

```rust
use rust_ai_agents_studio::api::{WsClient, WsState, WsMessage};

let mut client = WsClient::from_origin();

client.connect(
    |msg: WsMessage| {
        match msg {
            WsMessage::Metrics(m) => update_metrics(m),
            WsMessage::AgentStatus(s) => update_status(s),
            WsMessage::Trace(t) => add_trace(t),
        }
    },
    |state: WsState| {
        match state {
            WsState::Connected => log!("Connected"),
            WsState::Disconnected => log!("Disconnected"),
            WsState::Error(e) => log!("Error: {}", e),
        }
    },
)?;
```

## Global Context

Access WebSocket state from any component:

```rust
use rust_ai_agents_studio::WsContext;

#[component]
fn MyComponent() -> impl IntoView {
    let ws = use_context::<WsContext>().expect("WsContext not found");

    // Read connection state
    let is_connected = move || matches!(ws.state.get(), WsState::Connected);

    // Access last message
    let last_msg = ws.last_message;

    // Trigger reconnect
    let reconnect = move |_| {
        ws.connect.with_value(|f| f());
    };

    view! {
        <div>
            <span class:connected=is_connected>
                {move || if is_connected() { "Connected" } else { "Disconnected" }}
            </span>
            <button on:click=reconnect>"Reconnect"</button>
        </div>
    }
}
```

## Settings

Settings are persisted to localStorage:

```rust
#[derive(Serialize, Deserialize)]
struct Settings {
    // Connection
    api_url: String,
    ws_url: String,
    auto_connect: bool,

    // Display
    theme: Theme,
    refresh_interval: u32,
    max_traces: usize,

    // Features
    show_cost: bool,
    show_tokens: bool,
}
```

### Accessing Settings

```rust
use rust_ai_agents_studio::pages::settings::load_settings;

let settings = load_settings();
```

## Chat UI

Interactive chat with SSE streaming:

```rust
// The chat page handles:
// 1. Message input and history
// 2. SSE streaming from /api/chat/stream
// 3. Real-time token-by-token display
// 4. Error handling and retry
```

### Chat Features

- Message history
- SSE streaming responses
- Markdown rendering
- Code syntax highlighting
- Copy to clipboard
- Clear conversation
- Model selection

## Styling

The Studio uses Tailwind CSS classes:

```rust
view! {
    <div class="bg-gray-900 text-white p-4 rounded-lg">
        <h2 class="text-xl font-bold mb-2">"Title"</h2>
        <p class="text-gray-400">"Description"</p>
    </div>
}
```

## Development

### Hot Reload

```bash
cd crates/studio
trunk serve --open
```

### Debug Build

```bash
trunk build
```

### Production Build

```bash
trunk build --release
```

## Integration with Dashboard

The dashboard serves the Studio at `/studio`:

```rust
// In dashboard server.rs
.nest_service(
    "/studio/pkg",
    ServeDir::new("crates/dashboard/static/studio/pkg"),
)
```

### Deployment

1. Build Studio WASM:
   ```bash
   cd crates/studio && ./build.sh
   ```

2. Files are output to `crates/dashboard/static/studio/`

3. Dashboard serves at `http://localhost:3000/studio`

## File Structure

```
crates/studio/
├── src/
│   ├── lib.rs           # Main app, router, context
│   ├── api/
│   │   ├── mod.rs       # API exports
│   │   ├── client.rs    # REST client
│   │   └── websocket.rs # WebSocket client
│   ├── components/
│   │   ├── mod.rs       # Component exports
│   │   ├── layout.rs    # App layout
│   │   ├── navbar.rs    # Navigation bar
│   │   ├── sidebar.rs   # Side navigation
│   │   ├── cards.rs     # Stat/metric cards
│   │   ├── charts.rs    # SVG charts
│   │   └── tables.rs    # Data tables
│   └── pages/
│       ├── mod.rs       # Page exports
│       ├── home.rs      # Dashboard
│       ├── traces.rs    # Trace list
│       ├── sessions.rs  # Session list
│       ├── agents.rs    # Agent control
│       ├── metrics.rs   # Metrics detail
│       ├── chat.rs      # Chat UI
│       └── settings.rs  # Settings
├── index.html           # HTML template
├── Trunk.toml           # Trunk config
├── build.sh             # Build script
└── Cargo.toml
```

## Browser Support

- Chrome/Edge 88+
- Firefox 78+
- Safari 14+

Requires WebAssembly and WebSocket support.

## Related Crates

- [`rust-ai-agents-dashboard`](../dashboard) - Backend server
- [`rust-ai-agents-monitoring`](../monitoring) - Metrics tracking
- [`rust-ai-agents-agents`](../agents) - Agent implementation

## License

Apache-2.0
