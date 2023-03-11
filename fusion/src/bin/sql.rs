use std::env;
use std::path::Path;

use ballista::prelude::*;
use datafusion::prelude::CsvReadOptions;

#[tokio::main]
async fn main() -> Result<()> {
    let config = BallistaConfig::builder()
        .set("ballista.shuffle.partitions", "4")
        .build()?;
    let ctx = BallistaContext::remote("localhost", 50050, &config).await?;

    let homedir = env::var("GITHUB_DIR").unwrap();
    let csv_path = Path::new(&homedir)
        .join("haoxins/tiny-demos/fusion")
        .join("testdata/aggregate_test_100.csv");

    ctx.register_csv("test", csv_path.to_str().unwrap(), CsvReadOptions::new())
        .await?;

    let df = ctx
        .sql(
            "SELECT c1, MIN(c12), MAX(c12) \
        FROM test \
        WHERE c11 > 0.1 AND c11 < 0.9 \
        GROUP BY c1",
        )
        .await?;

    df.show().await?;

    Ok(())
}
