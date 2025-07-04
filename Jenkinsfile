pipeline {
    agent any
    
    environment {
        DOMINO_API_KEY = credentials('domino-api-key')
        DOMINO_URL = 'https://se-demo.domino.tech'
        DOMINO_PROJECT = 'jim_coates/NLP_Quality_Analytics'
    }
    
    stages {
        stage('Debug Info') {
            steps {
                script {
                    echo "🔍 Debug Information:"
                    echo "Branch Name: ${env.BRANCH_NAME ?: 'NOT SET'}"
                    echo "Git Branch: ${env.GIT_BRANCH ?: 'NOT SET'}"
                    echo "Build Number: ${BUILD_NUMBER}"
                    echo "Domino URL: ${DOMINO_URL}"
                    echo "Domino Project: ${DOMINO_PROJECT}"
                    echo "API URL will be: ${DOMINO_URL}/v1/projects/${DOMINO_PROJECT}/runs"
                }
            }
        }
        
        stage('Trigger Domino Training Job') {
            steps {
                script {
                    echo "🚀 Starting Domino job using legacy API..."
                    
                    // Based on legacy API docs - simple payload
                    def jobPayload = [
                        command: ["python", "experiments/mlflow_tracking.py"],
                        isDirect: false,
                        title: "Jenkins Triggered Training - Build ${BUILD_NUMBER}"
                    ]
                    
                    echo "📤 Sending payload: ${groovy.json.JsonOutput.toJson(jobPayload)}"
                    
                    def response = httpRequest(
                        httpMode: 'POST',
                        url: "${DOMINO_URL}/v1/projects/${DOMINO_PROJECT}/runs",
                        customHeaders: [
                            [name: 'X-Domino-Api-Key', value: DOMINO_API_KEY, maskValue: true],
                            [name: 'Content-Type', value: 'application/json']
                        ],
                        requestBody: groovy.json.JsonOutput.toJson(jobPayload),
                        validResponseCodes: '200:299'
                    )
                    
                    echo "📨 Response status: ${response.status}"
                    echo "📄 Response content: ${response.content}"
                    
                    // Extract runId from response content using simple string parsing
                    def responseText = response.content
                    if (responseText.contains('"runId"')) {
                        def runIdMatch = responseText =~ /"runId"\s*:\s*"([^"]+)"/
                        if (runIdMatch) {
                            env.DOMINO_RUN_ID = runIdMatch[0][1]
                            echo "✅ Started Domino run: ${env.DOMINO_RUN_ID}"
                        }
                    }
                    
                    echo "🔗 Check your Domino project for the new run!"
                }
            }
        }
        
        stage('Monitor Training Progress') {
            when {
                expression { env.DOMINO_RUN_ID != null }
            }
            steps {
                script {
                    echo "⏳ Monitoring training progress for run: ${env.DOMINO_RUN_ID}"
                    
                    timeout(time: 30, unit: 'MINUTES') {
                        waitUntil {
                            try {
                                def statusResponse = httpRequest(
                                    httpMode: 'GET',
                                    url: "${DOMINO_URL}/v1/projects/${DOMINO_PROJECT}/runs/${env.DOMINO_RUN_ID}",
                                    customHeaders: [
                                        [name: 'X-Domino-Api-Key', value: DOMINO_API_KEY, maskValue: true]
                                    ],
                                    validResponseCodes: '200:299'
                                )
                                
                                def responseText = statusResponse.content
                                echo "📊 Run status response: ${responseText}"
                                
                                // Simple string parsing to check status
                                def isFinished = responseText.contains('"status":"Succeeded"') || 
                                               responseText.contains('"status":"Failed"') || 
                                               responseText.contains('"status":"Error"') || 
                                               responseText.contains('"status":"Stopped"')
                                
                                if (isFinished) {
                                    echo "🏁 Run has finished!"
                                    return true
                                } else {
                                    echo "⏳ Run still in progress..."
                                    return false
                                }
                                
                            } catch (Exception e) {
                                echo "⚠️  Error checking run status: ${e.getMessage()}"
                                return true // Exit the wait loop on error
                            }
                        }
                    }
                }
            }
        }
        
        stage('Get Run Results') {
            when {
                expression { env.DOMINO_RUN_ID != null }
            }
            steps {
                script {
                    echo "📊 Fetching final run status for: ${env.DOMINO_RUN_ID}"
                    
                    try {
                        def finalResponse = httpRequest(
                            httpMode: 'GET',
                            url: "${DOMINO_URL}/v1/projects/${DOMINO_PROJECT}/runs/${env.DOMINO_RUN_ID}",
                            customHeaders: [
                                [name: 'X-Domino-Api-Key', value: DOMINO_API_KEY, maskValue: true]
                            ],
                            validResponseCodes: '200:299'
                        )
                        
                        def responseText = finalResponse.content
                        echo "🏁 Final run response: ${responseText}"
                        
                        if (responseText.contains('"status":"Succeeded"')) {
                            echo "🎉 Training job completed successfully!"
                        } else if (responseText.contains('"status":"Failed"')) {
                            echo "❌ Training job failed"
                            currentBuild.result = 'UNSTABLE'
                        } else {
                            echo "⚠️  Training job finished with unknown status"
                            currentBuild.result = 'UNSTABLE'
                        }
                        
                    } catch (Exception e) {
                        echo "⚠️  Could not get final run status: ${e.getMessage()}"
                    }
                }
            }
        }
    }
    
    post {
        always {
            echo "🏁 Pipeline completed"
        }
        success {
            script {
                echo "✅ Successfully started and monitored Domino training job!"
                if (env.DOMINO_RUN_ID) {
                    echo "🎯 Run ID: ${env.DOMINO_RUN_ID}"
                    echo "🔗 Check run details at: ${DOMINO_URL}/projects/${DOMINO_PROJECT}/runs"
                }
            }
        }
        failure {
            echo "❌ Pipeline failed"
        }
        unstable {
            echo "⚠️  Pipeline completed with warnings"
        }
    }
}