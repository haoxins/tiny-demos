        let saved_checksum = f.read_u32::<LittleEndian>()?;
        let key_len = f.read_u32::<LittleEndian>()?;
        let val_len = f.read_u32::<LittleEndian>()?;
        let data_len = key_len + val_len;
        let mut data = ByteString::with_capacity(data_len as usize);
        {
            f.by_ref().take(data_len as u64).read_to_end(&mut data)?;
        debug_assert_eq!(data.len(), data_len as usize);
        let checksum = CASTAGNOLI.checksum(&data);
        if checksum != saved_checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "checksum mismatch",
            ));
        let value = data.split_off(key_len as usize);
        let key = data;
        Ok(KeyValuePair { key, value })
    pub fn seek_to_end(&mut self) -> io::Result<u64> {
        self.f.seek(SeekFrom::End(0))
    pub fn load(&mut self) -> io::Result<()> {
        let mut f = BufReader::new(&self.f);
        loop {
            let current_position = f.seek(SeekFrom::Current(0))?;
            let maybe_kv = KV::process_record(&mut f);
            let kv = match maybe_kv {
                Ok(kv) => kv,
                Err(err) => match err.kind() {
                    io::ErrorKind::UnexpectedEof => {
                        break;
                    _ => return Err(err),
            self.index.insert(kv.key, current_position);
        Ok(())
    pub fn get(&mut self, key: &ByteStr) -> io::Result<Option<ByteString>> {
        let position = match self.index.get(key) {
            Some(position) => *position,
            None => return Ok(None),
        let kv = self.get_at(position)?;
        Ok(Some(kv.value))
    pub fn get_at(&mut self, position: u64) -> io::Result<KeyValuePair> {
        let mut f = BufReader::new(&mut self.f);
        f.seek(SeekFrom::Start(position))?;
        let kv = KV::process_record(&mut f)?;
        Ok(kv)
    }
    pub fn find(&mut self, target: &ByteStr) -> io::Result<Option<u64, ByteString>> {
        let mut f = BufReader::new(&mut self.f);
        let mut found: Option<(u64, ByteString)> = None;
        loop {
            let position = f.seek(SeekFrom::Current(0))?;
            let maybe_kv = KV::process_record(&mut f);
            let kv = match maybe_kv {
                Ok(kv) => kv,
                Err(err) => match err.kind() {
                    io::ErrorKind::UnexpectedEof => {
                        break;
                    }
                    _ => return Err(err),
            if kv.key == target {
                found = Some((position, kv.value));
        Ok(found)
    pub fn insert(&mut self, key: &ByteStr, value: &ByteStr) -> io::Result<()> {
        let position = self.insert_but_ignore_index(key, value)?;
        self.index.insert(key.to_vec(), position);
        Ok(())
    pub fn insert_but_ignore_index(&mut self, key: &ByteStr, value: &ByteStr) -> io::Result<u64> {
        let mut f = BufWriter::new(&mut self.f);
        let key_len = key.len();
        let val_len = value.len();
        let mut tmp = ByteString::with_capacity(key_len + val_len);
        for byte in key {
            tmp.push(*byte);
        }
        for byte in value {
            tmp.push(*byte);
        let checksum = CASTAGNOLI.checksum(&tmp);
        let next_byte = SeekFrom::End(0);
        let current_position = f.seek(SeekFrom::Current(0))?;
        f.seek(next_byte)?;
        f.write_u32::<LittleEndian>(checksum)?;
        f.write_u32::<LittleEndian>(key_len as u32)?;
        f.write_u32::<LittleEndian>(val_len as u32)?;
        f.write_all(&tmp)?;
        Ok(current_position)
    pub fn update(&mut self, key: &ByteStr, value: &ByteStr) -> io::Result<()> {
        self.insert(key, value)
    pub fn delete(&mut self, key: &ByteStr) -> io::Result<()> {
        self.insert(key, b"")
