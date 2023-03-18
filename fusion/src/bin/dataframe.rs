use fusion::util::get_file_path;

use ballista::prelude::*;
use datafusion::prelude::{col, lit, ParquetReadOptions};

#[tokio::main]
async fn main() -> Result<()> {
    let config = BallistaConfig::builder()
        .set("ballista.shuffle.partitions", "4")
        .build()?;
    let ctx = BallistaContext::remote("localhost", 50050, &config).await?;

    let parquet_path = get_file_path("alltypes_plain.parquet");

    let df = ctx
        .read_parquet(
            parquet_path.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?
        .select_columns(&["id", "bool_col", "timestamp_col"])?
        .filter(col("id").gt(lit(1)))?;

    df.show().await?;

    Ok(())
}
