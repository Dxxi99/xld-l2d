import SwiftUI

struct InspectorPanel: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            PanelHeader(title: "Inspector")
            Divider().background(Color(white: 0.25))

            // tab strip
            HStack(spacing: 0) {
                ForEach(AppState.InspectorTab.allCases) { tab in
                    Button {
                        appState.inspectorTab = tab
                    } label: {
                        Text(tab.rawValue)
                            .font(.system(size: 11, weight: appState.inspectorTab == tab ? .semibold : .regular))
                            .foregroundColor(appState.inspectorTab == tab ? .white : .white.opacity(0.55))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 6)
                            .background(appState.inspectorTab == tab ? Color(white: 0.20) : Color(white: 0.14))
                    }
                    .buttonStyle(.plain)
                }
            }
            Divider().background(Color(white: 0.25))

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    switch appState.inspectorTab {
                    case .properties: PropertiesTab()
                    case .behaviors:  PlaceholderTab(name: "Behaviors")
                    case .filters:    PlaceholderTab(name: "Filters")
                    case .object:     PlaceholderTab(name: "Object")
                    }
                }
                .padding(12)
            }
        }
    }
}

struct PropertiesTab: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Selection
            GroupBox {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Selection").font(.system(size: 10, weight: .semibold)).foregroundColor(.white.opacity(0.5))
                    Text(appState.selectedLayer ?? "None")
                        .font(.system(size: 12))
                        .foregroundColor(.white.opacity(0.9))
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(6)
            }
            .groupBoxStyle(InspectorGroupBox())

            // Pose
            GroupBox {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Pose").font(.system(size: 10, weight: .semibold)).foregroundColor(.white.opacity(0.5))

                    SliderRow(label: "Yaw",
                              value: $appState.yawDeg,
                              range: -30...30,
                              format: "%+.1f°")
                    SliderRow(label: "Pitch",
                              value: $appState.pitchDeg,
                              range: -15...15,
                              format: "%+.1f°")

                    Button("Reset") { appState.resetPose() }
                        .buttonStyle(.plain)
                        .font(.system(size: 11))
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background(RoundedRectangle(cornerRadius: 4).fill(Color(white: 0.22)))
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(6)
            }
            .groupBoxStyle(InspectorGroupBox())
        }
    }
}

struct SliderRow: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.system(size: 11)).foregroundColor(.white.opacity(0.7))
                Spacer()
                Text(String(format: format, value))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.white.opacity(0.9))
            }
            Slider(value: $value, in: range)
                .tint(.accentColor)
        }
    }
}

struct PlaceholderTab: View {
    let name: String
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(name).font(.system(size: 11, weight: .semibold)).foregroundColor(.white.opacity(0.5))
            Text("(empty)").font(.system(size: 11)).foregroundColor(.white.opacity(0.35))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct InspectorGroupBox: GroupBoxStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.content
            .background(RoundedRectangle(cornerRadius: 6).fill(Color(white: 0.16)))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(white: 0.24), lineWidth: 1))
    }
}
