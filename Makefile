CC = gcc
CFLAGS = -Wall -Wextra -O3 -fPIC
AR = ar rcs
TARGET = libndarray.a
TEST = test

SRC = ndarray.c
OBJ = $(SRC:.c=.o)

all: $(TARGET) $(TEST)

$(TARGET): $(OBJ)
	$(AR) $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST): test.c $(TARGET)
	$(CC) $(CFLAGS) -o $@ test.c -L. -lndarray -lm

clean:
	rm -f $(OBJ) $(TARGET) $(TEST)