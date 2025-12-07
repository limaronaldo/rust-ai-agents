//! WebSocket client for real-time updates

use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use web_sys::{CloseEvent, ErrorEvent, MessageEvent, WebSocket};

use super::types::WsMessage;

/// WebSocket connection state
#[derive(Debug, Clone, PartialEq)]
pub enum WsState {
    Connecting,
    Connected,
    Disconnected,
    Error(String),
}

/// WebSocket client for dashboard updates
pub struct WsClient {
    ws: Option<WebSocket>,
    url: String,
}

impl WsClient {
    /// Create a new WebSocket client
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            ws: None,
            url: url.into(),
        }
    }

    /// Create client pointing to current origin
    pub fn from_origin() -> Self {
        let window = web_sys::window().expect("no window");
        let location = window.location();
        let protocol = if location.protocol().unwrap_or_default() == "https:" {
            "wss:"
        } else {
            "ws:"
        };
        let host = location.host().unwrap_or_default();
        let url = format!("{}//{}/ws", protocol, host);
        Self::new(url)
    }

    /// Connect to the WebSocket
    pub fn connect<F, G>(&mut self, on_message: F, on_state_change: G) -> Result<(), String>
    where
        F: Fn(WsMessage) + 'static,
        G: Fn(WsState) + Clone + 'static,
    {
        let ws = WebSocket::new(&self.url).map_err(|e| format!("{:?}", e))?;

        // Set binary type
        ws.set_binary_type(web_sys::BinaryType::Arraybuffer);

        // On open
        let state_change = on_state_change.clone();
        let onopen = Closure::wrap(Box::new(move |_: JsValue| {
            state_change(WsState::Connected);
        }) as Box<dyn Fn(JsValue)>);
        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
        onopen.forget();

        // On message
        let onmessage = Closure::wrap(Box::new(move |e: MessageEvent| {
            if let Ok(text) = e.data().dyn_into::<js_sys::JsString>() {
                let text: String = text.into();
                if let Ok(msg) = serde_json::from_str::<WsMessage>(&text) {
                    on_message(msg);
                }
            }
        }) as Box<dyn Fn(MessageEvent)>);
        ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        onmessage.forget();

        // On error
        let state_change = on_state_change.clone();
        let onerror = Closure::wrap(Box::new(move |e: ErrorEvent| {
            state_change(WsState::Error(e.message()));
        }) as Box<dyn Fn(ErrorEvent)>);
        ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();

        // On close
        let onclose = Closure::wrap(Box::new(move |_: CloseEvent| {
            on_state_change(WsState::Disconnected);
        }) as Box<dyn Fn(CloseEvent)>);
        ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));
        onclose.forget();

        self.ws = Some(ws);
        Ok(())
    }

    /// Send a message
    pub fn send(&self, msg: &WsMessage) -> Result<(), String> {
        if let Some(ws) = &self.ws {
            let text = serde_json::to_string(msg).map_err(|e| e.to_string())?;
            ws.send_with_str(&text).map_err(|e| format!("{:?}", e))?;
        }
        Ok(())
    }

    /// Close the connection
    pub fn close(&self) {
        if let Some(ws) = &self.ws {
            let _ = ws.close();
        }
    }
}

/// Hook to use WebSocket connection in Leptos components
pub fn use_websocket() -> (
    ReadSignal<WsState>,
    ReadSignal<Option<WsMessage>>,
    impl Fn() + Clone,
) {
    let (state, set_state) = signal(WsState::Disconnected);
    let (message, set_message) = signal(None::<WsMessage>);

    let connect = move || {
        set_state.set(WsState::Connecting);

        let mut client = WsClient::from_origin();

        let _ = client.connect(
            move |msg| {
                set_message.set(Some(msg));
            },
            move |st| {
                set_state.set(st);
            },
        );
    };

    (state, message, connect)
}
