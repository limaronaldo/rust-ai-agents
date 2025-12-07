//! Simple chart components for metrics visualization
//! Uses SVG for rendering charts without external dependencies

use leptos::prelude::*;

/// Data point for time series
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub timestamp: f64,
    pub value: f64,
}

/// Line chart component
#[component]
pub fn LineChart(
    /// Data points to display
    data: Vec<DataPoint>,
    /// Chart width
    #[prop(default = 400)]
    width: u32,
    /// Chart height
    #[prop(default = 200)]
    height: u32,
    /// Line color
    #[prop(default = "#3b82f6")]
    color: &'static str,
    /// Show grid lines
    #[prop(default = true)]
    show_grid: bool,
    /// Y-axis label
    #[prop(optional)]
    y_label: Option<&'static str>,
) -> impl IntoView {
    let padding = 40.0;
    let chart_width = width as f64 - padding * 2.0;
    let chart_height = height as f64 - padding * 2.0;

    // Calculate min/max for scaling
    let (min_val, max_val) = if data.is_empty() {
        (0.0, 100.0)
    } else {
        let min = data.iter().map(|d| d.value).fold(f64::INFINITY, f64::min);
        let max = data
            .iter()
            .map(|d| d.value)
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        (min - range * 0.1, max + range * 0.1)
    };

    let (min_time, max_time) = if data.is_empty() {
        (0.0, 1.0)
    } else {
        (
            data.first().unwrap().timestamp,
            data.last().unwrap().timestamp,
        )
    };

    // Generate path
    let path = if data.is_empty() {
        String::new()
    } else {
        let time_range = max_time - min_time;
        let val_range = max_val - min_val;

        data.iter()
            .enumerate()
            .map(|(i, point)| {
                let x = if time_range > 0.0 {
                    padding + (point.timestamp - min_time) / time_range * chart_width
                } else {
                    padding + chart_width / 2.0
                };
                let y = if val_range > 0.0 {
                    padding + chart_height - (point.value - min_val) / val_range * chart_height
                } else {
                    padding + chart_height / 2.0
                };
                if i == 0 {
                    format!("M {} {}", x, y)
                } else {
                    format!("L {} {}", x, y)
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    };

    // Generate area fill path
    let area_path = if data.is_empty() || path.is_empty() {
        String::new()
    } else {
        let time_range = max_time - min_time;

        let first_x = if time_range > 0.0 {
            padding + (data.first().unwrap().timestamp - min_time) / time_range * chart_width
        } else {
            padding
        };
        let last_x = if time_range > 0.0 {
            padding + (data.last().unwrap().timestamp - min_time) / time_range * chart_width
        } else {
            padding + chart_width
        };
        let bottom_y = padding + chart_height;

        format!(
            "{} L {} {} L {} {} Z",
            path, last_x, bottom_y, first_x, bottom_y
        )
    };

    view! {
        <svg
            width=width
            height=height
            class="bg-gray-900 rounded-lg"
        >
            // Grid lines
            {show_grid.then(|| view! {
                <g class="text-gray-700">
                    // Horizontal grid lines
                    {(0..5).map(|i| {
                        let y = padding + (i as f64 / 4.0) * chart_height;
                        view! {
                            <line
                                x1=padding
                                y1=y
                                x2=padding + chart_width
                                y2=y
                                stroke="currentColor"
                                stroke-opacity="0.3"
                            />
                        }
                    }).collect::<Vec<_>>()}
                    // Vertical grid lines
                    {(0..5).map(|i| {
                        let x = padding + (i as f64 / 4.0) * chart_width;
                        view! {
                            <line
                                x1=x
                                y1=padding
                                x2=x
                                y2=padding + chart_height
                                stroke="currentColor"
                                stroke-opacity="0.3"
                            />
                        }
                    }).collect::<Vec<_>>()}
                </g>
            })}

            // Area fill
            {(!area_path.is_empty()).then(|| view! {
                <path
                    d=area_path.clone()
                    fill=color
                    fill-opacity="0.1"
                />
            })}

            // Line
            {(!path.is_empty()).then(|| view! {
                <path
                    d=path.clone()
                    fill="none"
                    stroke=color
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                />
            })}

            // Y-axis labels
            {(0..5).map(|i| {
                let y = padding + (i as f64 / 4.0) * chart_height;
                let val = max_val - (i as f64 / 4.0) * (max_val - min_val);
                view! {
                    <text
                        x=padding - 5.0
                        y=y + 4.0
                        text-anchor="end"
                        class="text-xs fill-gray-500"
                    >
                        {format!("{:.1}", val)}
                    </text>
                }
            }).collect::<Vec<_>>()}

            // Y-axis label
            {y_label.map(|label| view! {
                <text
                    x=12.0
                    y=height as f64 / 2.0
                    text-anchor="middle"
                    transform=format!("rotate(-90, 12, {})", height as f64 / 2.0)
                    class="text-xs fill-gray-400"
                >
                    {label}
                </text>
            })}

            // Empty state
            {data.is_empty().then(|| view! {
                <text
                    x=width as f64 / 2.0
                    y=height as f64 / 2.0
                    text-anchor="middle"
                    class="text-sm fill-gray-500"
                >
                    "No data"
                </text>
            })}
        </svg>
    }
}

/// Bar chart component
#[component]
pub fn BarChart(
    /// Labels for each bar
    labels: Vec<String>,
    /// Values for each bar
    values: Vec<f64>,
    /// Chart width
    #[prop(default = 400)]
    width: u32,
    /// Chart height
    #[prop(default = 200)]
    height: u32,
    /// Bar color
    #[prop(default = "#3b82f6")]
    color: &'static str,
) -> impl IntoView {
    let padding = 40.0;
    let chart_width = width as f64 - padding * 2.0;
    let chart_height = height as f64 - padding * 2.0;

    let max_val = values.iter().fold(0.0f64, |a, &b| a.max(b));
    let max_val = if max_val == 0.0 { 1.0 } else { max_val * 1.1 };

    let bar_count = values.len().max(1);
    let bar_width = (chart_width / bar_count as f64) * 0.7;
    let bar_gap = (chart_width / bar_count as f64) * 0.3;

    view! {
        <svg
            width=width
            height=height
            class="bg-gray-900 rounded-lg"
        >
            // Bars
            {values.iter().enumerate().map(|(i, &value)| {
                let bar_height = (value / max_val) * chart_height;
                let x = padding + (i as f64 * (bar_width + bar_gap)) + bar_gap / 2.0;
                let y = padding + chart_height - bar_height;
                let label = labels.get(i).cloned().unwrap_or_default();

                view! {
                    <g>
                        <rect
                            x=x
                            y=y
                            width=bar_width
                            height=bar_height
                            fill=color
                            rx="4"
                        />
                        // Label
                        <text
                            x=x + bar_width / 2.0
                            y=padding + chart_height + 15.0
                            text-anchor="middle"
                            class="text-xs fill-gray-400"
                        >
                            {label}
                        </text>
                        // Value
                        <text
                            x=x + bar_width / 2.0
                            y=y - 5.0
                            text-anchor="middle"
                            class="text-xs fill-gray-300"
                        >
                            {format!("{:.0}", value)}
                        </text>
                    </g>
                }
            }).collect::<Vec<_>>()}

            // Empty state
            {values.is_empty().then(|| view! {
                <text
                    x=width as f64 / 2.0
                    y=height as f64 / 2.0
                    text-anchor="middle"
                    class="text-sm fill-gray-500"
                >
                    "No data"
                </text>
            })}
        </svg>
    }
}

/// Stat card with sparkline
#[component]
pub fn SparklineCard(
    /// Card title
    title: &'static str,
    /// Current value
    value: String,
    /// Historical data points (just values, timestamps auto-generated)
    history: Vec<f64>,
    /// Trend direction (optional)
    #[prop(optional)]
    trend: Option<Trend>,
    /// Sparkline color
    #[prop(default = "#3b82f6")]
    color: &'static str,
) -> impl IntoView {
    let data: Vec<DataPoint> = history
        .iter()
        .enumerate()
        .map(|(i, &v)| DataPoint {
            timestamp: i as f64,
            value: v,
        })
        .collect();

    view! {
        <div class="bg-gray-800 rounded-lg p-4">
            <div class="flex items-center justify-between mb-2">
                <p class="text-sm text-gray-400">{title}</p>
                {trend.map(|t| {
                    let (icon, color) = match t {
                        Trend::Up => ("↑", "text-green-400"),
                        Trend::Down => ("↓", "text-red-400"),
                        Trend::Stable => ("→", "text-gray-400"),
                    };
                    view! { <span class=color>{icon}</span> }
                })}
            </div>
            <p class="text-2xl font-bold text-white mb-2">{value}</p>
            <LineChart
                data=data
                width=200
                height=60
                color=color
                show_grid=false
            />
        </div>
    }
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Trend {
    Up,
    Down,
    Stable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_point_creation() {
        let point = DataPoint {
            timestamp: 1.0,
            value: 100.5,
        };

        assert_eq!(point.timestamp, 1.0);
        assert!((point.value - 100.5).abs() < 0.001);
    }

    #[test]
    fn test_data_point_series() {
        let series: Vec<DataPoint> = (0..10)
            .map(|i| DataPoint {
                timestamp: i as f64,
                value: (i * 10) as f64,
            })
            .collect();

        assert_eq!(series.len(), 10);
        assert_eq!(series[0].value, 0.0);
        assert_eq!(series[9].value, 90.0);
    }

    #[test]
    fn test_trend_equality() {
        assert_eq!(Trend::Up, Trend::Up);
        assert_eq!(Trend::Down, Trend::Down);
        assert_eq!(Trend::Stable, Trend::Stable);
        assert_ne!(Trend::Up, Trend::Down);
    }

    #[test]
    fn test_trend_clone_copy() {
        let trend = Trend::Up;
        let cloned = trend.clone();
        let copied: Trend = trend;

        assert_eq!(trend, cloned);
        assert_eq!(trend, copied);
    }

    #[test]
    fn test_calculate_trend() {
        fn calc_trend(data: &[f64]) -> Option<Trend> {
            if data.len() < 2 {
                return None;
            }
            let last = *data.last()?;
            let prev = *data.get(data.len() - 2)?;
            Some(if last > prev {
                Trend::Up
            } else if last < prev {
                Trend::Down
            } else {
                Trend::Stable
            })
        }

        assert_eq!(calc_trend(&[1.0, 2.0, 3.0]), Some(Trend::Up));
        assert_eq!(calc_trend(&[3.0, 2.0, 1.0]), Some(Trend::Down));
        assert_eq!(calc_trend(&[1.0, 1.0, 1.0]), Some(Trend::Stable));
        assert_eq!(calc_trend(&[1.0]), None);
        assert_eq!(calc_trend(&[]), None);
    }

    #[test]
    fn test_data_point_min_max() {
        let data = vec![
            DataPoint {
                timestamp: 0.0,
                value: 10.0,
            },
            DataPoint {
                timestamp: 1.0,
                value: 5.0,
            },
            DataPoint {
                timestamp: 2.0,
                value: 15.0,
            },
        ];

        let min = data.iter().map(|d| d.value).fold(f64::INFINITY, f64::min);
        let max = data
            .iter()
            .map(|d| d.value)
            .fold(f64::NEG_INFINITY, f64::max);

        assert!((min - 5.0).abs() < 0.001);
        assert!((max - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_value() {
        fn normalize(value: f64, min: f64, max: f64) -> f64 {
            if (max - min).abs() < f64::EPSILON {
                0.5
            } else {
                (value - min) / (max - min)
            }
        }

        assert!((normalize(50.0, 0.0, 100.0) - 0.5).abs() < 0.001);
        assert!((normalize(0.0, 0.0, 100.0) - 0.0).abs() < 0.001);
        assert!((normalize(100.0, 0.0, 100.0) - 1.0).abs() < 0.001);
        assert!((normalize(50.0, 50.0, 50.0) - 0.5).abs() < 0.001);
    }
}

/// Gauge/meter component
#[component]
pub fn Gauge(
    /// Current value (0-100)
    value: f64,
    /// Gauge label
    label: &'static str,
    /// Size in pixels
    #[prop(default = 120)]
    size: u32,
) -> impl IntoView {
    let radius = size as f64 / 2.0 - 10.0;
    let circumference = 2.0 * std::f64::consts::PI * radius;
    let progress = (value.clamp(0.0, 100.0) / 100.0) * 0.75; // 270 degrees
    let stroke_dasharray = format!("{} {}", circumference * progress, circumference);

    let color = if value < 33.0 {
        "#22c55e" // green
    } else if value < 66.0 {
        "#eab308" // yellow
    } else {
        "#ef4444" // red
    };

    view! {
        <div class="flex flex-col items-center">
            <svg width=size height=size class="transform -rotate-135">
                // Background arc
                <circle
                    cx=size as f64 / 2.0
                    cy=size as f64 / 2.0
                    r=radius
                    fill="none"
                    stroke="#374151"
                    stroke-width="8"
                    stroke-dasharray=format!("{} {}", circumference * 0.75, circumference)
                    stroke-linecap="round"
                />
                // Progress arc
                <circle
                    cx=size as f64 / 2.0
                    cy=size as f64 / 2.0
                    r=radius
                    fill="none"
                    stroke=color
                    stroke-width="8"
                    stroke-dasharray=stroke_dasharray
                    stroke-linecap="round"
                />
            </svg>
            <div class="absolute flex flex-col items-center justify-center" style=format!("width: {}px; height: {}px;", size, size)>
                <span class="text-2xl font-bold text-white">{format!("{:.0}%", value)}</span>
            </div>
            <p class="text-sm text-gray-400 mt-2">{label}</p>
        </div>
    }
}
