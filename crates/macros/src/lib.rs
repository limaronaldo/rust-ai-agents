//! # Procedural Macros for rust-ai-agents
//!
//! This crate provides derive macros for easily defining tools and other
//! agent components with minimal boilerplate.
//!
//! ## Tool Derive Macro
//!
//! ```rust,ignore
//! use rust_ai_agents_macros::Tool;
//!
//! #[derive(Tool)]
//! #[tool(name = "calculator", description = "Perform mathematical calculations")]
//! struct CalculatorTool {
//!     #[tool(param, required, description = "The mathematical expression to evaluate")]
//!     expression: String,
//!
//!     #[tool(param, description = "Number of decimal places for result")]
//!     precision: Option<u32>,
//! }
//!
//! impl CalculatorTool {
//!     async fn run(&self, expression: String, precision: Option<u32>) -> Result<serde_json::Value, String> {
//!         // Implementation here
//!         Ok(serde_json::json!(42.0))
//!     }
//! }
//! ```

use darling::{ast, FromDeriveInput, FromField};
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Ident, Type};

/// Attributes for the Tool derive macro at struct level
#[derive(Debug, FromDeriveInput)]
#[darling(attributes(tool), supports(struct_named))]
struct ToolArgs {
    ident: Ident,
    data: ast::Data<(), ToolField>,

    /// Tool name (defaults to snake_case of struct name)
    #[darling(default)]
    name: Option<String>,

    /// Tool description
    #[darling(default)]
    description: Option<String>,

    /// Whether the tool is dangerous and requires confirmation
    #[darling(default)]
    dangerous: bool,
}

/// Attributes for tool parameters (struct fields)
#[derive(Debug, FromField)]
#[darling(attributes(tool))]
struct ToolField {
    ident: Option<Ident>,
    ty: Type,

    /// Mark this field as a tool parameter
    #[darling(default)]
    param: bool,

    /// Whether this parameter is required
    #[darling(default)]
    required: bool,

    /// Parameter description
    #[darling(default)]
    description: Option<String>,

    /// Skip this field (not a parameter)
    #[darling(default)]
    skip: bool,
}

/// Derive macro for creating Tool implementations
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Tool)]
/// #[tool(name = "search", description = "Search the web")]
/// struct SearchTool {
///     #[tool(param, required, description = "Search query")]
///     query: String,
///
///     #[tool(param, description = "Maximum results to return")]
///     max_results: Option<u32>,
/// }
///
/// impl SearchTool {
///     // You must implement this method
///     async fn run(&self, query: String, max_results: Option<u32>) -> Result<serde_json::Value, String> {
///         Ok(serde_json::json!({"results": []}))
///     }
/// }
/// ```
#[proc_macro_derive(Tool, attributes(tool))]
pub fn derive_tool(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let args = match ToolArgs::from_derive_input(&input) {
        Ok(args) => args,
        Err(e) => return e.write_errors().into(),
    };

    let expanded = generate_tool_impl(&args);

    TokenStream::from(expanded)
}

fn generate_tool_impl(args: &ToolArgs) -> proc_macro2::TokenStream {
    let struct_name = &args.ident;

    // Generate tool name (default to snake_case of struct name without "Tool" suffix)
    let tool_name = args.name.clone().unwrap_or_else(|| {
        let name = struct_name.to_string();
        let name = name.strip_suffix("Tool").unwrap_or(&name);
        to_snake_case(name)
    });

    let description = args
        .description
        .clone()
        .unwrap_or_else(|| format!("{} tool", tool_name));

    let dangerous = args.dangerous;

    // Extract parameter fields
    let fields = match &args.data {
        ast::Data::Struct(fields) => fields,
        _ => panic!("Tool derive only supports structs"),
    };

    let params: Vec<_> = fields
        .fields
        .iter()
        .filter(|f| f.param && !f.skip)
        .collect();

    // Generate JSON schema for parameters
    let param_properties = generate_param_properties(&params);
    let required_params = generate_required_params(&params);

    // Generate argument extraction code
    let arg_extractions = generate_arg_extractions(&params);

    // Generate parameter names for calling run()
    let param_names: Vec<_> = params.iter().map(|f| f.ident.as_ref().unwrap()).collect();

    quote! {
        #[async_trait::async_trait]
        impl rust_ai_agents_core::tool::Tool for #struct_name {
            fn schema(&self) -> rust_ai_agents_core::tool::ToolSchema {
                rust_ai_agents_core::tool::ToolSchema {
                    name: #tool_name.to_string(),
                    description: #description.to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            #(#param_properties),*
                        },
                        "required": [#(#required_params),*]
                    }),
                    dangerous: #dangerous,
                    metadata: std::collections::HashMap::new(),
                }
            }

            async fn execute(
                &self,
                _context: &rust_ai_agents_core::tool::ExecutionContext,
                arguments: serde_json::Value,
            ) -> Result<serde_json::Value, rust_ai_agents_core::errors::ToolError> {
                #(#arg_extractions)*

                self.run(#(#param_names),*).await
                    .map_err(|e| rust_ai_agents_core::errors::ToolError::ExecutionFailed(e.to_string()))
            }
        }
    }
}

fn generate_param_properties(params: &[&ToolField]) -> Vec<proc_macro2::TokenStream> {
    params
        .iter()
        .map(|field| {
            let name = field.ident.as_ref().unwrap().to_string();
            let description = field
                .description
                .clone()
                .unwrap_or_else(|| format!("Parameter: {}", name));
            let json_type = type_to_json_type(&field.ty);

            quote! {
                #name: {
                    "type": #json_type,
                    "description": #description
                }
            }
        })
        .collect()
}

fn generate_required_params(params: &[&ToolField]) -> Vec<proc_macro2::TokenStream> {
    params
        .iter()
        .filter(|f| f.required)
        .map(|field| {
            let name = field.ident.as_ref().unwrap().to_string();
            quote! { #name }
        })
        .collect()
}

fn generate_arg_extractions(params: &[&ToolField]) -> Vec<proc_macro2::TokenStream> {
    params
        .iter()
        .map(|field| {
            let ident = field.ident.as_ref().unwrap();
            let name = ident.to_string();
            let ty = &field.ty;

            if is_option_type(ty) {
                quote! {
                    let #ident: #ty = arguments.get(#name)
                        .and_then(|v| serde_json::from_value(v.clone()).ok());
                }
            } else if field.required {
                quote! {
                    let #ident: #ty = {
                        let val = arguments.get(#name)
                            .ok_or_else(|| rust_ai_agents_core::errors::ToolError::InvalidArguments(
                                format!("Missing required parameter: {}", #name)
                            ))?
                            .clone();
                        serde_json::from_value(val)
                            .map_err(|e| rust_ai_agents_core::errors::ToolError::InvalidArguments(
                                format!("Invalid type for {}: {}", #name, e)
                            ))?
                    };
                }
            } else {
                // Non-required, non-option field - use default
                quote! {
                    let #ident: #ty = arguments.get(#name)
                        .and_then(|v| serde_json::from_value(v.clone()).ok())
                        .unwrap_or_default();
                }
            }
        })
        .collect()
}

fn type_to_json_type(ty: &Type) -> &'static str {
    let ty_str = quote!(#ty).to_string();
    let ty_str = ty_str.replace(' ', "");

    if ty_str.contains("String") || ty_str.contains("&str") {
        "string"
    } else if ty_str.contains("i8")
        || ty_str.contains("i16")
        || ty_str.contains("i32")
        || ty_str.contains("i64")
        || ty_str.contains("u8")
        || ty_str.contains("u16")
        || ty_str.contains("u32")
        || ty_str.contains("u64")
        || ty_str.contains("isize")
        || ty_str.contains("usize")
    {
        "integer"
    } else if ty_str.contains("f32") || ty_str.contains("f64") {
        "number"
    } else if ty_str.contains("bool") {
        "boolean"
    } else if ty_str.contains("Vec") || ty_str.contains("Array") {
        "array"
    } else {
        "object"
    }
}

fn is_option_type(ty: &Type) -> bool {
    let ty_str = quote!(#ty).to_string();
    ty_str.contains("Option")
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}
