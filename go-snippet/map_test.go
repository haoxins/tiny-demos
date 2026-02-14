package snippet

import "testing"

func TestMapNil(t *testing.T) {
	var m map[string]int
	m = nil
	if _, ok := m["404"]; !ok {
		t.Log("map is nil, key not found as expected")
	}
}
