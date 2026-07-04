//! Argument-level rejection. Messages are model-presentable and always
//! include a literal example of valid input — small models correct
//! themselves from examples, not from prose.

export class ArgError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ArgError";
  }
}
