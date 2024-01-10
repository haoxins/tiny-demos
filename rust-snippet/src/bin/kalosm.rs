use std::io::Write;

use kalosm::{language::*, *};

#[tokio::main]
async fn main() {
    let mut llm = Phi::start().await;
    let prompt = "The following is a 300 word essay about Paris:";
    print!("{prompt}");

    let stream = llm.stream_text(prompt).with_max_length(1000).await.unwrap();

    let mut sentences = stream.words();
    while let Some(text) = sentences.next().await {
        print!("{text}");
        std::io::stdout().flush().unwrap();
    }
}
