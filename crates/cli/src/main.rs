use anyhow::Result;
use clap::{Parser, Subcommand};
use config::{AppConfig, TCPSubjectSelectionConfig, load_config};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "masters", version, about = "Preprocess + Process Pipeline")]
struct Cli {
    /// Path to a config TOML file
    #[arg(long, global = true, default_value = "config.toml")]
    config: PathBuf,

    /// Logging filter, e.g. 'info', 'debug', 'trace', 'myproj=debug'
    #[arg(long, global = true, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    TCPSelectSubjects {
        #[arg(long)]
        tcp_dir: Option<PathBuf>,

        #[arg(long)]
        output_dir: Option<PathBuf>,

        #[arg(long, value_delimiter = ',')]
        filters: Option<Vec<String>>,

        #[arg(long)]
        dry_run: Option<bool>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(cli.log_level.clone()))
        .init();

    let cfg = load_config(&cli.config).unwrap_or_else(|_| AppConfig {
        tcp_subject_selection: TCPSubjectSelectionConfig::default(),
    });

    match cli.cmd {
        Command::TCPSelectSubjects {
            tcp_dir,
            output_dir,
            filters,
            dry_run,
        } => {
            // I/O config
            let mut p = cfg.tcp_subject_selection;

            if let Some(v) = tcp_dir {
                p.tcp_dir = v
            };
            if let Some(v) = output_dir {
                p.output_dir = v
            };

            // Optional filters
            if let Some(ref v) = filters
                && !v.is_empty()
            {
                p.filters = filters;
            }

            // Run options
            if let Some(v) = dry_run {
                p.dry_run = v
            };

            tcp_subject_selection::run(&p)
        }
    }
}
