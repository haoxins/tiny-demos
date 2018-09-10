
package hello;

import java.util.Optional;

public class Hello {
  public static void main(String[] args) {

    User.findById("404").ifPresentOrElse(user -> {
      System.out.println(user);
    }, () -> {
      System.out.println("null");
    });

    User.findUsers().stream().filter(user -> {
      return user.ifPresent();
    });
  }
}
