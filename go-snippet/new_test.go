package snippet

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNew(t *testing.T) {
	type User struct {
		Name *string
		Age  *int
	}

	u := User{
		Name: new("Alice"),
		Age:  new(30),
	}

	assert.Equal(t, "Alice", *u.Name)
	assert.Equal(t, 30, *u.Age)

	u.Name = nil
	assert.Nil(t, u.Name)

	u.Name = new("Xin")
	assert.Equal(t, "Xin", *u.Name)
}
