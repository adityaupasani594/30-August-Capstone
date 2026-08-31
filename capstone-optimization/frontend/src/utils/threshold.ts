export function calculateThreshold(
  capacity: number,
  _speed: number,
  _length: number,
  roadType: string,
  isReference?: boolean,
  networkType?: string
): number {
  if (networkType === "aditya") {
    return 0.60 * capacity;
  }

  if (isReference || roadType.toLowerCase() === "reference") {
    return 0;
  }

  const type = roadType.toLowerCase();
  if (type === "arterial") {
    return 0.85 * capacity;
  } else if (type === "expressway") {
    return 0.95 * capacity;
  }
  return 0;
}
