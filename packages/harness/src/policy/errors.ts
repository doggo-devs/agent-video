/**
 * Argument-level rejection error.
 * Messages are model-presentable and carry a literal example of valid input.
 */

export class ArgError extends Error {
	constructor(message: string) {
		super(message)
		this.name = 'ArgError'
	}
}
