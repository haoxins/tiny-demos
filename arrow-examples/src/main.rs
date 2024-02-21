use ballista::prelude::*;
use datafusion::prelude::{avg, col, max, min, sum, ParquetReadOptions};

#[tokio::main]
async fn main() -> Result<()> {
    // create configuration
    let config = BallistaConfig::builder()
        .set("ballista.shuffle.partitions", "4")
        .build()?;

    // connect to Ballista scheduler
    let ctx = BallistaContext::remote("localhost", 50050, &config).await?;

    let filename = "testdata/yellow_tripdata_2022-01.parquet";

    // define the query using the DataFrame trait
    let df = ctx
        .read_parquet(filename, ParquetReadOptions::default())
        .await?
        .select_columns(&["passenger_count", "fare_amount"])?
        .aggregate(
            vec![col("passenger_count")],
            vec![
                min(col("fare_amount")),
                max(col("fare_amount")),
                avg(col("fare_amount")),
                sum(col("fare_amount")),
            ],
        )?
        .sort(vec![col("passenger_count").sort(true, true)])?;

    // this is equivalent to the following SQL
    // SELECT passenger_count, MIN(fare_amount), MAX(fare_amount), AVG(fare_amount), SUM(fare_amount)
    //     FROM tripdata
    //     GROUP BY passenger_count
    //     ORDER BY passenger_count

    // print the results
    df.show().await?;

    Ok(())
}
