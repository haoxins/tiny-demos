package hello;

import java.util.Optional;
import java.util.ArrayList;
import java.util.List;

class User {
  public static Optional<String> findById(String id) {
    String user = null;

    if (id == "404") {
      return Optional.ofNullable(user);
    }

    return Optional.ofNullable("User: " + id);
  }
}
