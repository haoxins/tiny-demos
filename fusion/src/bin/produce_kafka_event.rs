use rdkafka::config::ClientConfig;
use rdkafka::producer::{BaseProducer, BaseRecord, Producer};
use std::time::Duration;

/*
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Event",
  "description": "",
  "type": "object",
  "properties": {
    "amount": {
      "description": "",
      "type": "integer"
    }
  }
}
*/

fn main() {
    let producer: BaseProducer = ClientConfig::new()
        .set("bootstrap.servers", "localhost:9092")
        .create()
        .unwrap();

    const TOPIC_NAME: &str = "test";
    const MESSAGE: &str = r#"{"amount": 666}"#;

    println!("Sending message: {}", MESSAGE);

    producer
        .send(BaseRecord::to(TOPIC_NAME).payload(MESSAGE).key("key"))
        .unwrap();

    let _ = producer.flush(Duration::from_secs(1));
}
