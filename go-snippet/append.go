package snippet

import "testing"

func Test_append_a_nil_slice(t *testing.T) {
	var a []int
	a = append(a, 1)
	var b []int = nil
	a = append(a, b...)
}
