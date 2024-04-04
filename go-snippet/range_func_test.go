package snippet

import (
	"fmt"
	"testing"

	_ "github.com/stretchr/testify/assert"
)

func TestRange(t *testing.T) {
	for i := range 10 {
		fmt.Println(10 - i)
	}

	list := []string{"hello", "world"}
	for k := range list {
		fmt.Println(list[k])
	}
}
