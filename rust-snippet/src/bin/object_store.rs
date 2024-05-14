use std::env;
use std::fs;
use std::path::Path;

use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::path::Path as ObjectStorePath;
use object_store::ObjectStore;

#[tokio::main]
async fn main() {
    let bucket_name = env::var("BUCKET_NAME").unwrap();
    let src_path = env::var("SRC_PATH").unwrap();
    let dst_path = env::var("DST_PATH").unwrap();

    let client = GoogleCloudStorageBuilder::from_env()
        .with_bucket_name(bucket_name)
        .build()
        .unwrap();

    let object = client
        .get(&ObjectStorePath::parse(src_path).unwrap())
        .await
        .unwrap();

    let data = object.bytes().await.unwrap();
    fs::write(Path::new(&dst_path), data).unwrap();
}
