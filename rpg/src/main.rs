trait Enchanter: std::fmt::Debug {
    fn competency(&self) -> f64;

    fn enchant(&self, thing: &mut Thing) {
        let probability_of_success = self.competency();
        let spell_is_successful = rand::thread_rng().gen_bool(probability_of_success);

        print!("{:?} mutters incoherently.", self);

        if spell_is_successful {
            println!("The {:?} glows brightly.", thing);
        } else {
            println!(
                "The {:?} fizzes, then turns into a worthless trinket.",
                thing
            );
            *thing = Thing::Trinket {};
        }
    }
}
