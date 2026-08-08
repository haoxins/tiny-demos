package snippet

import "testing"

func Test_range(t *testing.T) {
	for range 0 {
		t.Fatal("won't run")
	}
	for range -1 {
		t.Fatal("won't run")
	}
	a := [...]int{3, 2, 1}
	for i := range a {
		t.Log("range", i)
	}
	for range a {
		t.Log("_")
	}
}
