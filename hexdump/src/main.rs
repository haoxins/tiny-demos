    let mut f = File::open(&fname).expect("Unable to open file");
    let mut buffer = [0; BYTES_PER_LINE];
    while let Ok(_) = f.read_exact(&mut buffer) {
        print!("[0x{:08x}] ", pos);
        for byte in &buffer {
            match *byte {
                0x00 => print!(". "),
                0xff => print!("## "),
                _ => print!("{:02x} ", byte),
