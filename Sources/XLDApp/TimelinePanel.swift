import SwiftUI

struct TimelinePanel: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // header
            HStack {
                Text("Timeline")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.white.opacity(0.7))

                Divider().frame(height: 12)

                // playback controls
                HStack(spacing: 8) {
                    Image(systemName: "backward.end.fill")
                    Image(systemName: "play.fill")
                    Image(systemName: "forward.end.fill")
                }
                .font(.system(size: 10))
                .foregroundColor(.white.opacity(0.6))

                Text("00:00:00:00")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.white.opacity(0.8))

                Spacer()

                Text("30 fps")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.white.opacity(0.5))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(Color(white: 0.14))

            Divider().background(Color(white: 0.25))

            // placeholder track rows
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    TrackRow(name: "Head",   keys: [0.2, 0.5, 0.8])
                    TrackRow(name: "Eye_L",  keys: [0.3, 0.6])
                    TrackRow(name: "Eye_R",  keys: [0.3, 0.6])
                    TrackRow(name: "Hair_F", keys: [0.1, 0.4, 0.7, 0.9])
                    TrackRow(name: "Body",   keys: [0.5])
                }
            }
        }
    }
}

struct TrackRow: View {
    let name: String
    let keys: [Double]   // 0..1 positions

    var body: some View {
        HStack(spacing: 0) {
            Text(name)
                .font(.system(size: 11))
                .foregroundColor(.white.opacity(0.7))
                .frame(width: 120, alignment: .leading)
                .padding(.leading, 10)

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Rectangle().fill(Color(white: 0.13))
                    ForEach(keys.indices, id: \.self) { i in
                        Circle()
                            .fill(Color.accentColor)
                            .frame(width: 7, height: 7)
                            .offset(x: CGFloat(keys[i]) * geo.size.width - 3.5)
                    }
                }
            }
            .frame(height: 22)
            .padding(.vertical, 2)
        }
    }
}
