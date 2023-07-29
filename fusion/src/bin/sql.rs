use std::env;

use ballista::prelude::*;
use datafusion::prelude::ParquetReadOptions;

use fusion::util::read_sql;

#[tokio::main]
async fn main() -> Result<()> {
    let config = BallistaConfig::builder()
        .set("ballista.shuffle.partitions", "4")
        .build()?;
    let ctx = BallistaContext::remote("localhost", 50050, &config).await?;

    let parquet_dir = env::var("PARQUET_DIR").unwrap();
    ctx.register_parquet("bw", parquet_dir.as_str(), ParquetReadOptions::default())
        .await?;

    let sql_1 = read_sql("sql_1");

    let df = ctx.sql(sql_1.as_str()).await?;

    df.show().await?;

    Ok(())
}
