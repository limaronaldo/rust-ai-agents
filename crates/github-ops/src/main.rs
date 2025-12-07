use anyhow::{Context, Result};
use clap::Parser;
use colored::*;
use serde::Deserialize;
use std::path::PathBuf;
use tokio::process::Command;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the JSON configuration file
    #[arg(short, long, default_value = "scripts/create_github_issues.json")]
    config: PathBuf,

    /// GitHub owner (user or organization)
    #[arg(short, long)]
    owner: String,

    /// GitHub repository name
    #[arg(short, long)]
    repo: String,

    /// Dry run (do not create actual issues)
    #[arg(long)]
    dry_run: bool,
}

#[derive(Deserialize, Debug)]
struct Config {
    milestone: Milestone,
    labels: Vec<Label>,
    issues: Vec<IssueConfig>,
}

#[derive(Deserialize, Debug)]
struct Milestone {
    title: String,
    description: String,
    due_on: String,
}

#[derive(Deserialize, Debug)]
struct Label {
    name: String,
    color: String,
    description: String,
}

#[derive(Deserialize, Debug)]
struct IssueConfig {
    title: String,
    labels: Vec<String>,
    #[serde(rename = "body_path")]
    body_path: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!(
        "{}",
        format!("🚀 GitHub Ops: {}/{}", args.owner, args.repo)
            .bold()
            .green()
    );
    if args.dry_run {
        println!(
            "{}",
            "⚠️  DRY RUN MODE: No changes will be applied".yellow()
        );
    }
    println!();

    // 1. Read Config
    let config_content = std::fs::read_to_string(&args.config)
        .context(format!("Failed to read config file: {:?}", args.config))?;
    let config: Config =
        serde_json::from_str(&config_content).context("Failed to parse JSON config")?;

    println!("📄 Loaded config with {} issues", config.issues.len());

    // 2. Check Prerequisites (gh CLI)
    if !check_command("gh", &["--version"]).await? {
        anyhow::bail!("'gh' CLI is not installed or not in PATH");
    }
    if !check_command("gh", &["auth", "status"]).await? {
        anyhow::bail!("'gh' CLI is not authenticated. Run 'gh auth login'");
    }

    // 3. Create Labels
    println!("\n🏷️  Labels");
    for label in &config.labels {
        create_label(&args, label).await?;
    }

    // 4. Create Milestone
    println!("\n🚩 Milestone");
    let milestone_number = create_milestone(&args, &config.milestone).await?;
    println!("   Using milestone number: {}", milestone_number);

    // 5. Create Issues
    println!("\n📝 Issues");
    for (i, issue) in config.issues.iter().enumerate() {
        create_issue(
            &args,
            issue,
            milestone_number,
            i + 1,
            config.issues.len(),
            &config.milestone.title,
        )
        .await?;
    }

    println!(
        "\n{}",
        "✨ All operations completed successfully!".green().bold()
    );
    Ok(())
}

async fn check_command(cmd: &str, args: &[&str]) -> Result<bool> {
    let status = Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await;
    Ok(status.map(|s| s.success()).unwrap_or(false))
}

async fn create_label(args: &Args, label: &Label) -> Result<()> {
    print!("   Creating label '{}'...", label.name.cyan());

    if args.dry_run {
        println!(" {}", "[Skipped]".yellow());
        return Ok(());
    }

    let output = Command::new("gh")
        .args(&[
            "label",
            "create",
            &label.name,
            "--color",
            &label.color,
            "--description",
            &label.description,
            "--repo",
            &format!("{}/{}", args.owner, args.repo),
            "--force", // Updates if exists
        ])
        .output()
        .await?;

    if output.status.success() {
        println!(" {}", "OK".green());
    } else {
        // gh label create fails if it exists, even with --force sometimes depending on version,
        // but let's try to edit if create fails
        let str_err = String::from_utf8_lossy(&output.stderr);
        if str_err.contains("already exists") {
            let edit_output = Command::new("gh")
                .args(&[
                    "label",
                    "edit",
                    &label.name,
                    "--color",
                    &label.color,
                    "--description",
                    &label.description,
                    "--repo",
                    &format!("{}/{}", args.owner, args.repo),
                ])
                .output()
                .await?;

            if edit_output.status.success() {
                println!(" {}", "Updated".blue());
            } else {
                println!(" {}", "Failed (ignored)".red());
            }
        } else {
            println!(" {}", "Failed".red());
        }
    }
    Ok(())
}

async fn create_milestone(args: &Args, milestone: &Milestone) -> Result<u64> {
    // Check existence
    let check_cmd = Command::new("gh")
        .args(&[
            "api",
            &format!("repos/{}/{}/milestones", args.owner, args.repo),
            "--jq",
            &format!(".[] | select(.title == \"{}\") | .number", milestone.title),
        ])
        .output()
        .await?;

    let existing_number = String::from_utf8_lossy(&check_cmd.stdout)
        .trim()
        .to_string();

    if !existing_number.is_empty() {
        return Ok(existing_number.parse().unwrap_or(0));
    }

    if args.dry_run {
        println!("   Would create milestone '{}'", milestone.title);
        return Ok(0);
    }

    let create_cmd = Command::new("gh")
        .args(&[
            "api",
            &format!("repos/{}/{}/milestones", args.owner, args.repo),
            "-X",
            "POST",
            "-f",
            &format!("title={}", milestone.title),
            "-f",
            &format!("description={}", milestone.description),
            "-f",
            &format!("due_on={}", milestone.due_on),
            "--jq",
            ".number",
        ])
        .output()
        .await?;

    let number = String::from_utf8_lossy(&create_cmd.stdout)
        .trim()
        .to_string();
    Ok(number.parse().unwrap_or(0))
}

async fn create_issue(
    args: &Args,
    issue: &IssueConfig,
    _milestone_number: u64,
    index: usize,
    total: usize,
    milestone_title: &str, // Fallback for dry-run display
) -> Result<()> {
    print!("   [{}/{}] '{}'...", index, total, issue.title.cyan());

    // Load body
    let mut body_path = args
        .config
        .parent()
        .unwrap_or(&PathBuf::from("."))
        .join(&issue.body_path);
    if !body_path.exists() {
        // Try relative to cwd
        body_path = issue.body_path.clone();
    }

    // Check existence
    let check_cmd = Command::new("gh")
        .args(&[
            "issue",
            "list",
            "--repo",
            &format!("{}/{}", args.owner, args.repo),
            "--search",
            &format!("in:title \"{}\"", issue.title),
            "--json",
            "number",
            "--jq",
            ".[0].number",
        ])
        .output()
        .await?;

    let existing = String::from_utf8_lossy(&check_cmd.stdout)
        .trim()
        .to_string();

    if !existing.is_empty() && existing != "null" {
        println!(" {} (#{})", "Exists".yellow(), existing);
        return Ok(());
    }

    if args.dry_run {
        println!(" {}", "[Skipped]".yellow());
        return Ok(());
    }

    let mut cmd = Command::new("gh");
    cmd.args(&[
        "issue",
        "create",
        "--repo",
        &format!("{}/{}", args.owner, args.repo),
        "--title",
        &issue.title,
        "--body-file",
        body_path.to_str().unwrap(),
        "--milestone",
        &milestone_title, // gh issue create accepts title or number
    ]);

    for label in &issue.labels {
        cmd.args(&["--label", label]);
    }

    let output = cmd.output().await?;

    if output.status.success() {
        let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
        println!(" {} ({})", "Created".green(), url);
    } else {
        println!(" {}", "Failed".red());
        println!("{}", String::from_utf8_lossy(&output.stderr));
    }

    Ok(())
}
