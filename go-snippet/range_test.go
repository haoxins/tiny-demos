package snippet

import "testing"

func TestRange(t *testing.T) {
	for i := range -3 {
		t.Log("range", i)
	}
	a := [...]int{3, 2, 1}
	for i := range a {
		t.Log("range", i)
	}
	for range a {
		t.Log("_")
	}
}
