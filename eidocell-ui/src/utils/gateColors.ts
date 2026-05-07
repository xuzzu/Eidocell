export const GATE_COLORS = [
  '#E53E3E', '#DD6B20', '#D69E2E', '#38A169',
  '#319795', '#3182CE', '#5A67D8', '#805AD5',
  '#D53F8C', '#00B5D8', '#ED64A6', '#9333EA',
] as const

let colorIndex = 0

export function nextGateColor(): string {
  const color = GATE_COLORS[colorIndex % GATE_COLORS.length]
  colorIndex += 1
  return color
}
