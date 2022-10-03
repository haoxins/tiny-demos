    let args: Vec<String> = std::env::args().collect();
    let fname = args.get(1).expect(&USAGE);
    let cmd = args.get(2).expect(&USAGE).as_ref();
    let key = args.get(3).expect(&USAGE).as_ref();
    let value = args.get(4);
    let path = std::path::Path::new(&fname);
    let mut store = KV::open(path).expect("Failed to open file");
    store.load().expect("Failed to load data");
    match cmd {
        "get" => match store.get(key).unwrap() {
            None => eprintln!("{:?} not found", key),
            Some(value) => println!("{:?}", value),
        },
        "delete" => store.delete(key).unwrap(),
        "insert" => {
            let value = value.expect(&USAGE).as_ref();
            store.insert(key, value).unwrap()
        }
        "update" => {
            let value = value.expect(&USAGE).as_ref();
            store.update(key, value).unwrap()
        }
        _ => eprintln!("{}", USAGE),
