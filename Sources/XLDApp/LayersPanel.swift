import SwiftUI

struct LayersPanel: View {
    @EnvironmentObject var appState: AppState

    // Placeholder layer tree (matching Motion's style)
    private let tree: [LayerNode] = [
        .group("Character", [
            .group("Head", [
                .leaf("Face"),
                .leaf("Eye_L"),
                .leaf("Eye_R"),
                .leaf("Eyebrow_L"),
                .leaf("Eyebrow_R"),
                .leaf("Mouth"),
                .leaf("Hair_Front"),
                .leaf("Hair_Back"),
            ]),
            .group("Body", [
                .leaf("Neck"),
                .leaf("Clothes"),
                .leaf("Arm_L"),
                .leaf("Arm_R"),
            ]),
        ])
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            PanelHeader(title: "Layers")
            Divider().background(Color(white: 0.25))

            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(tree) { node in
                        LayerRow(node: node, depth: 0, selected: $appState.selectedLayer)
                    }
                }
                .padding(.vertical, 6)
                .padding(.horizontal, 4)
            }
        }
    }
}

struct PanelHeader: View {
    let title: String
    var body: some View {
        HStack {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.white.opacity(0.7))
            Spacer()
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color(white: 0.14))
    }
}

indirect enum LayerNode: Identifiable {
    case group(String, [LayerNode])
    case leaf(String)
    var id: String {
        switch self {
        case .group(let n, _): return "G:\(n)"
        case .leaf(let n): return "L:\(n)"
        }
    }
    var name: String {
        switch self {
        case .group(let n, _): return n
        case .leaf(let n): return n
        }
    }
    var isGroup: Bool {
        if case .group = self { return true }
        return false
    }
    var children: [LayerNode] {
        if case .group(_, let c) = self { return c }
        return []
    }
}

struct LayerRow: View {
    let node: LayerNode
    let depth: Int
    @Binding var selected: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                Spacer().frame(width: CGFloat(depth) * 12)
                Image(systemName: node.isGroup ? "folder" : "square.on.square")
                    .font(.system(size: 10))
                    .foregroundColor(.white.opacity(0.5))
                Text(node.name)
                    .font(.system(size: 12))
                    .foregroundColor(selected == node.name ? .white : .white.opacity(0.75))
                Spacer()
            }
            .padding(.vertical, 3)
            .padding(.horizontal, 6)
            .background(selected == node.name ? Color.accentColor.opacity(0.35) : Color.clear)
            .contentShape(Rectangle())
            .onTapGesture { selected = node.name }

            ForEach(node.children) { child in
                LayerRow(node: child, depth: depth + 1, selected: $selected)
            }
        }
    }
}
