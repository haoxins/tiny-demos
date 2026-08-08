package snippet

import "testing"

func Test_append(t *testing.T) {
	var a []int
	a = append(a, 1)
	var b []int = nil
	a = append(a, b...)
}
